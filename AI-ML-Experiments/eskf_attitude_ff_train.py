# MARID ESKF Attitude Feedforward Training
"""
Train a feedforward network to predict yaw correction from the ESKF estimate.

- Input  (12-D): ESKF state   [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
- Output  (1-D): Δyaw = wrap(yaw_gt - yaw_eskf)

Roll and pitch are intentionally input-only. They are already well observed by the
raw-IMU tilt path; training the feedforward model to correct them risks fighting a
good physical/sensor estimate. Yaw is the weak, outlier-prone attitude axis.

WHY correction instead of absolute yaw:
  Absolute yaw is circular and mission-heading dependent. The useful supervised
  target is the local yaw error after wrapping to [-π, π]. That lets the model learn
  when the ESKF yaw is biased without asking it to memorize global heading.

Modes (TRAIN_PER_FLIGHT):
  False — train one model on all flights combined (cross-flight generalisation).
  True  — train one model per flight independently. Each flight's distribution
          dominates its own normalisation; the model learns attitude/velocity
          patterns specific to that flight's conditions.
          Models saved as eskf_attitude_ff_<flight_id>.pt.

Data split strategy — per-flight temporal split (not random):
  Random 80/20 across time-series produces val samples that are ~20 ms from train
  samples — not genuine generalisation. Instead, for each flight we use the first
  80% of timesteps as train and the last 20% as val.

Data source: eskf_gt_logger  (keys: eskf_inputs, pose_targets, flight_id)
"""
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 4096   # mini-batch size; full-batch GD on 1M+ rows causes OOM on CPU

DATA_FOLDER = 'data_sync'       # 'data' → 15-col base;  'data_low_err' → enriched 23-col corpus
DATA_DIR    = Path(f'~/marid_ws/{DATA_FOLDER}').expanduser()

SELECTED_FLIGHTS    = []
EXCLUDED_FLIGHTS    = []
TRAIN_PER_FLIGHT    = False  # True → one model per flight; False → one model for all
EARLY_STOP_PATIENCE = 100
VAL_MSE_TARGET      = 0.0
INCLUDE_MIRRORED    = True   # include logger-saved _mirror flights in train (never in val)
VAL_FLIGHTS         = [                            # cold-start held-out flights — identical to velocity LSTM val set
    'flight_20260531_075734',
    'flight_20260601_082638',
]
_VAL_TIMESTAMPS = [v.replace('flight_', '') for v in VAL_FLIGHTS]  # for prefix-agnostic mirror exclusion

LABELS = ['Δyaw (rad, wrapped)']
MIN_FLIGHT_STEPS = 400   # need at least some data for a meaningful split

# ── Data loading ──────────────────────────────────────────────────────────────

_G        = 9.81
_MIN_V    = 3.0
_PDOT_MAX = 2.0

def _augment_eskf_inputs(eskf: np.ndarray, d) -> np.ndarray:
    """Augment the 12-col ESKF input.

    data mode        → 15 cols: base(12) + thrust + ground_flag + psi_dot_aero
    data_low_err    → 23 cols: above(15) + imu_acc(3) + a_excess + delta_yaw_madgwick + airspeed
                                + delta_yaw_sun*sun_valid + sun_valid

    imu_acc preprocessing:
      ay clipped to ±15 m/s²  (Gazebo physics spikes)
      az gravity-subtracted   az_aero = az - g·cos(roll)·cos(pitch)  (removes redundant attitude-gravity term)
      a_excess = |a_raw| - g  (load-factor deviation; computed before gravity subtraction)
    """
    N           = len(eskf)
    thrust      = d['thrust'].astype(np.float32).ravel()[:N] if 'thrust' in d else np.zeros(N, np.float32)
    z           = eskf[:, 2]
    roll        = eskf[:, 3]
    pitch       = eskf[:, 4]
    yaw_eskf    = eskf[:, 5]
    vx, vy      = eskf[:, 6], eskf[:, 7]
    ground_flag = (z < 1.0).astype(np.float32)

    if DATA_FOLDER in ('data_low_err', 'data_sync') and 'airspeed' in d:
        V_src = np.clip(np.abs(d['airspeed'].astype(np.float32).ravel()[:N]), _MIN_V, None)
    else:
        V_src = np.clip(np.sqrt(vx**2 + vy**2), _MIN_V, None)

    psi_dot = np.clip((_G / V_src) * np.tan(roll), -_PDOT_MAX, _PDOT_MAX).astype(np.float32)
    psi_dot *= (1.0 - ground_flag)

    base = np.concatenate([eskf, thrust[:, None], ground_flag[:, None], psi_dot[:, None]], axis=1)

    if DATA_FOLDER not in ('data_low_err', 'data_sync') or 'imu_acc' not in d:
        return base   # 15 cols

    def _wrap(a): return ((a + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)

    imu_acc      = d['imu_acc'].astype(np.float32)[:N]
    imu_acc[:, 1] = np.clip(imu_acc[:, 1], -15.0, 15.0)  # Gazebo ay spikes: ±1.5g lateral impossible in coordinated flight
    a_excess = (np.linalg.norm(imu_acc, axis=1) - _G).astype(np.float32)  # load-factor deviation, computed before gravity subtraction
    imu_acc[:, 2] -= (_G * np.cos(roll) * np.cos(pitch)).astype(np.float32)  # remove gravity projection → net aerodynamic normal load
    yaw_madgwick = d['yaw_madgwick'].astype(np.float32).ravel()[:N]
    airspeed     = d['airspeed'].astype(np.float32).ravel()[:N]
    sun_yaw      = d['sun_yaw'].astype(np.float32).ravel()[:N]
    sun_valid    = d['sun_valid'].astype(np.float32).ravel()[:N]

    delta_madgwick = _wrap(yaw_madgwick - yaw_eskf)
    delta_sun      = _wrap(sun_yaw      - yaw_eskf) * sun_valid

    return np.concatenate([base,
                           imu_acc,
                           a_excess[:, None],
                           delta_madgwick[:, None],
                           airspeed[:, None],
                           delta_sun[:, None],
                           sun_valid[:, None]], axis=1)   # 23 cols

def _wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _load_targets(d):
    """Return 1-D wrapped yaw correction: Δyaw = wrap(yaw_gt - yaw_eskf).

      eskf_inputs yaw is always column 5.
      9-D legacy pose_targets yaw:  col 5 [x,y,z,roll,pitch,yaw,vx,vy,vz]
      7-D current pose_targets yaw: col 4 [x,y,roll,pitch,yaw,vx,vy]
    """
    X = d['eskf_inputs'].astype(np.float32)
    y = d['pose_targets'].astype(np.float32)
    yaw_gt = y[:, 5] if y.shape[1] == 9 else y[:, 4]
    yaw_eskf = X[:, 5]
    dyaw = _wrap_pi(yaw_gt - yaw_eskf).astype(np.float32)
    return dyaw[:, None]

try:
    file_list = list(uploaded.keys())  # Colab
    flight_groups = {'colab': sorted(file_list)}
    legacy_files  = []
except NameError:
    all_files = sorted(DATA_DIR.glob("marid_eskf_gt_*.npz"))
    if not all_files:
        raise FileNotFoundError(
            f"No marid_eskf_gt_*.npz files found in {DATA_DIR}. "
            "Run eskf_gt_logger during a flight first."
        )
    flight_groups = defaultdict(list)
    legacy_files  = []
    for p in all_files:
        d = np.load(p, allow_pickle=True)
        if 'flight_id' in d:
            fid = str(d['flight_id'])
            flight_groups[fid].append(p)
        else:
            legacy_files.append(p)

    for fid in flight_groups:
        flight_groups[fid].sort()

    if SELECTED_FLIGHTS:
        flight_groups = {fid: chunks for fid, chunks in flight_groups.items()
                         if fid in SELECTED_FLIGHTS}
        legacy_files  = []
    if EXCLUDED_FLIGHTS:
        flight_groups = {fid: chunks for fid, chunks in flight_groups.items()
                         if fid not in EXCLUDED_FLIGHTS}

print(f'Found {len(flight_groups)} new-format flight(s), {len(legacy_files)} legacy chunk(s)')

# ── Model ─────────────────────────────────────────────────────────────────────

INPUT_DIM = 23 if DATA_FOLDER in ('data_low_err', 'data_sync') else 15

def make_model():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 1),
    )

# ── Training function ─────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, label=''):
    model     = make_model()
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100)

    epochs          = 2000
    train_losses    = []
    val_losses      = []
    best_val_loss   = float('inf')
    best_epoch      = 0
    best_state_dict = None

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True)

    print(f'\nStarting training{" — " + label if label else ""}...', flush=True)

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_train_loss = float(np.mean(batch_losses))
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            v_loss = loss_fn(model(X_val), y_val)
        val_losses.append(v_loss.item())
        scheduler.step(v_loss)

        multiplicity = ' *** NEW VALIDATION BEST ***' if best_epoch == epoch + 1 else f' ---> No Improvement for {epoch - best_epoch + 2} epochs <---'

        if v_loss.item() < best_val_loss:
            best_val_loss   = v_loss.item()
            best_epoch      = epoch + 1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs}  Train MSE: {epoch_train_loss:.6f}  '
                  f'Val MSE: {v_loss.item():.6f}  Best val: {best_val_loss:.6f} @ epoch {best_epoch}' + multiplicity, flush=True)

        if best_val_loss <= VAL_MSE_TARGET:
            print(f'\nTarget val MSE {VAL_MSE_TARGET} reached at epoch {epoch+1}.')
            break

        if epoch + 1 - best_epoch >= EARLY_STOP_PATIENCE:
            print(f'\nEarly stop at epoch {epoch+1} — no val improvement for {EARLY_STOP_PATIENCE} epochs.')
            break

    print(f'\nRestoring best model from epoch {best_epoch} (val MSE {best_val_loss:.6f})')
    model.load_state_dict(best_state_dict)
    return model, train_losses, val_losses, best_val_loss

def eval_rmse(model, X_val_raw, y_val_raw, X_mean, X_std, y_mean, y_std):
    model.eval()
    X_val_norm = torch.tensor((X_val_raw - X_mean) / X_std)
    with torch.no_grad():
        y_pred_norm = model(X_val_norm).numpy()
    y_pred_phys = y_pred_norm * y_std + y_mean
    return [float(np.sqrt(np.mean((y_val_raw[:, i] - y_pred_phys[:, i])**2)))
            for i in range(len(LABELS))]

def plot_losses(train_losses, val_losses, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ax1.plot(train_losses, 'r-', label='Train MSE', linewidth=0.8)
    ax1.plot(val_losses,   'b-', label='Val MSE',   linewidth=0.8)
    ax1.set_ylabel('MSE'); ax1.set_yscale('log'); ax1.legend(); ax1.grid()
    ax1.set_title(title)
    start = len(train_losses) // 2
    ax2.plot(range(start, len(train_losses)), train_losses[start:], 'r-', linewidth=0.8, label='Train MSE')
    ax2.plot(range(start, len(val_losses)),   val_losses[start:],   'b-', linewidth=0.8, label='Val MSE')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE'); ax2.legend(); ax2.grid()
    ax2.set_title('Zoom: second half')
    plt.tight_layout()
    plt.show()

def load_flight(chunk_paths):
    X_parts, y_parts = [], []
    for p in chunk_paths:
        d = np.load(p, allow_pickle=True)
        X_parts.append(_augment_eskf_inputs(d['eskf_inputs'].astype(np.float32), d))
        y_parts.append(_load_targets(d))
    return np.concatenate(X_parts), np.concatenate(y_parts)

# ── Per-flight training ───────────────────────────────────────────────────────

if TRAIN_PER_FLIGHT:
    summary = []

    for fid, chunk_paths in sorted(flight_groups.items()):
        X_raw, y_raw = load_flight(chunk_paths)
        N = len(X_raw)

        if N < MIN_FLIGHT_STEPS:
            print(f'\n── {fid}: {N} steps — too short, skipping ──')
            continue

        split = int(0.8 * N)
        print(f'\n{"─"*60}')
        print(f'Flight: {fid}  ({N} steps → {split} train / {N-split} val)')
        print(f'{"─"*60}')

        X_train_raw, y_train_raw = X_raw[:split], y_raw[:split]
        X_val_raw,   y_val_raw   = X_raw[split:], y_raw[split:]

        X_mean = X_train_raw.mean(axis=0); X_std = X_train_raw.std(axis=0); X_std[X_std < 1e-8] = 1.0
        y_mean = y_train_raw.mean(axis=0); y_std = y_train_raw.std(axis=0); y_std[y_std < 1e-8] = 1.0

        X_train = torch.tensor((X_train_raw - X_mean) / X_std)
        y_train = torch.tensor((y_train_raw - y_mean) / y_std)
        X_val   = torch.tensor((X_val_raw   - X_mean) / X_std)
        y_val   = torch.tensor((y_val_raw   - y_mean) / y_std)

        model, tl, vl, best_val = train_model(X_train, y_train, X_val, y_val, label=fid)
        plot_losses(tl, vl, f'ESKF Attitude FF — {fid}')

        rmse = eval_rmse(model, X_val_raw, y_val_raw, X_mean, X_std, y_mean, y_std)
        print(f'\nPer-output validation RMSE — {fid}:')
        for lbl, r in zip(LABELS, rmse):
            print(f'  {lbl:14s}: RMSE = {r:.4f}')
        summary.append((fid, best_val, rmse))

        pt_path   = DATA_DIR / f'eskf_attitude_ff_{fid}.pt'
        norm_path = DATA_DIR / f'eskf_attitude_ff_norm_{fid}.npz'
        torch.save(model.state_dict(), pt_path)
        np.savez(norm_path, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
        print(f'Saved {pt_path.name}')

    if summary:
        print(f'\n{"═"*60}')
        print('PER-FLIGHT SUMMARY — eskf_attitude_ff')
        print(f'{"═"*60}')
        hdr = f'{"Flight":<30}  {"Val MSE":>8}  ' + '  '.join(f'{l[:8]:>8}' for l in LABELS)
        print(hdr)
        for fid, bval, rmse in summary:
            row = f'{fid:<30}  {bval:>8.4f}  ' + '  '.join(f'{r:>8.3f}' for r in rmse)
            print(row)

        results = {fid: {'val_mse': bval, 'rmse': dict(zip(LABELS, rmse))}
                   for fid, bval, rmse in summary}
        out = DATA_DIR / 'results_eskf_attitude_ff.json'
        out.write_text(json.dumps(results, indent=2))
        print(f'\nResults saved to {out.name}')

# ── Cross-flight training (original mode) ─────────────────────────────────────

else:
    X_train_list, y_train_list = [], []
    X_val_list,   y_val_list   = [], []
    val_flights = []   # [(fid, X_val_fid, y_val_fid)] for per-flight JSON

    def _load_chunk(p):
        d = np.load(p, allow_pickle=True)
        return _augment_eskf_inputs(d['eskf_inputs'].astype(np.float32), d), _load_targets(d)

    def _split_flight(fid, chunk_paths, n_chunks):
        n_val = max(1, round(0.2 * n_chunks))
        val_set = set(chunk_paths[-n_val:])
        print(f'  {fid}: {n_chunks} chunks → {n_chunks - n_val} train, {n_val} val')
        X_vf, y_vf = [], []
        for p in chunk_paths:
            X, y = _load_chunk(p)
            if p in val_set:
                X_val_list.append(X); y_val_list.append(y)
                X_vf.append(X);       y_vf.append(y)
            else:
                X_train_list.append(X); y_train_list.append(y)
        if X_vf:
            val_flights.append((fid, np.concatenate(X_vf), np.concatenate(y_vf)))

    for p in legacy_files:
        X, y = _load_chunk(p)
        X_train_list.append(X)
        y_train_list.append(y)

    for fid, chunk_paths in flight_groups.items():
        is_mirror = '_mirror' in fid
        if is_mirror and not INCLUDE_MIRRORED:
            continue
        if VAL_FLIGHTS and any(ts in fid and '_mirror' in fid for ts in _VAL_TIMESTAMPS):
            print(f'  {fid}: excluded (mirror of val flight)')
            continue
        n = len(chunk_paths)
        if VAL_FLIGHTS and fid in VAL_FLIGHTS:
            X_vf, y_vf = [], []
            for p in chunk_paths:
                X, y = _load_chunk(p)
                X_vf.append(X); y_vf.append(y)
            X_vf_cat = np.concatenate(X_vf); y_vf_cat = np.concatenate(y_vf)
            X_val_list.append(X_vf_cat); y_val_list.append(y_vf_cat)
            val_flights.append((fid, X_vf_cat, y_vf_cat))
            print(f'  {fid}: {n} chunk(s) → VAL FLIGHT (held out entirely)')
        elif is_mirror or n < 2:
            for p in chunk_paths:
                X, y = _load_chunk(p)
                X_train_list.append(X); y_train_list.append(y)
            tag = 'mirror, train only' if is_mirror else 'train only'
            print(f'  {fid}: {n} chunk(s) → {tag}')
        else:
            _split_flight(fid, chunk_paths, n)

    X_train_raw = np.concatenate(X_train_list); y_train_raw = np.concatenate(y_train_list)
    X_val_raw   = np.concatenate(X_val_list);   y_val_raw   = np.concatenate(y_val_list)
    print(f'\nTrain: X {X_train_raw.shape}, y {y_train_raw.shape}')
    print(f'Val:   X {X_val_raw.shape},   y {y_val_raw.shape}')

    X_mean = X_train_raw.mean(axis=0); X_std = X_train_raw.std(axis=0); X_std[X_std < 1e-8] = 1.0
    y_mean = y_train_raw.mean(axis=0); y_std = y_train_raw.std(axis=0); y_std[y_std < 1e-8] = 1.0

    X_train = torch.tensor((X_train_raw - X_mean) / X_std)
    y_train = torch.tensor((y_train_raw - y_mean) / y_std)
    X_val   = torch.tensor((X_val_raw   - X_mean) / X_std)
    y_val   = torch.tensor((y_val_raw   - y_mean) / y_std)

    model, tl, vl, _ = train_model(X_train, y_train, X_val, y_val)
    plot_losses(tl, vl, 'ESKF Attitude FF — all flights')

    rmse = eval_rmse(model, X_val_raw, y_val_raw, X_mean, X_std, y_mean, y_std)
    print('\nPer-output validation RMSE (physical units):')
    for lbl, r in zip(LABELS, rmse):
        print(f'  {lbl:14s}: RMSE = {r:.4f}')

    torch.save(model.state_dict(), DATA_DIR / 'eskf_attitude_ff.pt')
    np.savez(DATA_DIR / 'eskf_attitude_ff_norm.npz',
             X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    print(f'\nSaved model and normalization to {DATA_DIR}')

    # ── Per-flight eval → JSON for compare_results.py ────────────────────────
    cf_results = {}
    for fid, X_vf, y_vf in val_flights:
        rmse_f = eval_rmse(model, X_vf, y_vf, X_mean, X_std, y_mean, y_std)
        fmse   = float(np.mean([r**2 for r in rmse_f]))
        cf_results[fid] = {'val_mse': fmse, 'rmse': dict(zip(LABELS, rmse_f))}
        print(f'  {fid}  val_mse={fmse:.4f}  ' +
              '  '.join(f'{l}={r:.3f}' for l, r in zip(LABELS, rmse_f)))
    if cf_results:
        out = DATA_DIR / 'results_eskf_attitude_ff.json'
        out.write_text(json.dumps(cf_results, indent=2))
        print(f'Results saved to {out.name}')
