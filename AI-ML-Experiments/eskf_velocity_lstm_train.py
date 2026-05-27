"""
MARID ESKF Velocity LSTM Training
Trains an LSTM to predict velocity corrections (Δvx, Δvy) by processing each flight
as one continuous sequence with hidden state carried forward from takeoff.

- Input  (T, 12): ESKF state sequence [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
- Output (T, 2):  corrections at every timestep [Δvx, Δvy] = [vx_gt - vx_est, vy_gt - vy_est]

Why velocity rather than position:
  Δvx/Δvy is bounded (typically ±5 m/s), locally observable from pitch/roll/airspeed,
  and causally linked to aerodynamic state. Δx/Δy grows without bound and requires
  predicting the full integral of velocity error — much harder to generalise from few flights.

  At inference, the ESKF applies Δvx/Δvy corrections directly to its velocity state.
  Position correction follows automatically via integration.

Modes (TRAIN_PER_FLIGHT):
  False — train one model on all flights combined (cross-flight generalisation).
  True  — train one model per flight independently.

Training: Truncated BPTT (TBPTT) — process flight in CHUNK_LEN chunks,
  carry hidden state forward between chunks, detach gradients at chunk boundaries.

Val: warm up hidden state on train portion (first 80%), then evaluate on val portion
  (last 20%) — hidden state is correctly initialised, not cold-started.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

DATA_FOLDER = 'data_extended'   # 'data' → 15-col base;  'data_extended' → 24-col rich
DATA_DIR  = Path(f'~/marid_ws/{DATA_FOLDER}').expanduser()
CHUNK_LEN = 300  # TBPTT chunk length (6s at 50 Hz)

SELECTED_FLIGHTS    = []
EXCLUDED_FLIGHTS    = []
TRAIN_PER_FLIGHT    = False
EARLY_STOP_PATIENCE = 100
VAL_MSE_TARGET      = 0.0
INCLUDE_MIRRORED    = True   # include logger-saved _mirror flights in train (never in val)
VAL_FLIGHTS         = [                            # cold-start held-out flights (never seen during training)
    'flight_20260521_211150',                      # oldest session — temporal shift; 2D spread; 8.9 min; yaw 4.9°
    'flight_20260522_062211',                      # mid-period — 2D spread (2546×2365 m); 14.6 min; yaw 4.8°
    'flight_20260523_090404',                      # longest flight (29.1 min); max drift stress; yaw 3.7°
]

LABELS = ['Δvx (m/s)', 'Δvy (m/s)']
MIN_FLIGHT_STEPS = 2 * CHUNK_LEN

# ── Data loading ──────────────────────────────────────────────────────────────

_G        = 9.81
_MIN_V    = 3.0   # m/s — matches coordinated-turn gate; avoids div-by-zero at low speed
_PDOT_MAX = 2.0   # rad/s — cap psi_dot_aero at extreme bank / near-zero speed

def _augment_eskf_inputs(eskf: np.ndarray, d) -> np.ndarray:
    """Augment the 12-col ESKF input.

    data mode        → 15 cols: base(12) + thrust + ground_flag + psi_dot_aero
    data_extended    → 23 cols: above(15) + imu_acc(3) + a_excess + yaw_madgwick + airspeed
                                + delta_yaw_sun*sun_valid + sun_valid

    imu_acc preprocessing:
      ay clipped to ±15 m/s²  (Gazebo physics spikes)
      az gravity-subtracted   az_aero = az - g·cos(roll)·cos(pitch)  (removes redundant attitude-gravity term)
      a_excess = |a_raw| - g  (load-factor deviation; computed before gravity subtraction)

    Extended yaw features are stored as DELTAS from yaw_eskf (wrapped to [-π,π]),
    zeroed when the source is invalid — bounded, frame-independent, directly represent
    how far each reference has drifted from the ESKF estimate.
    """
    N           = len(eskf)
    thrust      = d['thrust'].astype(np.float32).ravel()[:N] if 'thrust' in d else np.zeros(N, np.float32)
    z           = eskf[:, 2]
    roll        = eskf[:, 3]
    pitch       = eskf[:, 4]
    yaw_eskf    = eskf[:, 5]
    vx, vy      = eskf[:, 6], eskf[:, 7]
    ground_flag = (z < 1.0).astype(np.float32)

    if DATA_FOLDER == 'data_extended' and 'airspeed' in d:
        V_src = np.clip(np.abs(d['airspeed'].astype(np.float32).ravel()[:N]), _MIN_V, None)
    else:
        V_src = np.clip(np.sqrt(vx**2 + vy**2), _MIN_V, None)

    psi_dot = np.clip((_G / V_src) * np.tan(roll), -_PDOT_MAX, _PDOT_MAX).astype(np.float32)
    psi_dot *= (1.0 - ground_flag)

    base = np.concatenate([eskf, thrust[:, None], ground_flag[:, None], psi_dot[:, None]], axis=1)

    if DATA_FOLDER != 'data_extended' or 'imu_acc' not in d:
        return base   # 15 cols

    def _wrap(a): return ((a + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)

    imu_acc      = d['imu_acc'].astype(np.float32)[:N]            # (N, 3)
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

def _load_vxy(d):
    """Return [Δvx, Δvy] = GT world-frame velocity minus ESKF estimate."""
    eskf = d['eskf_inputs'].astype(np.float32)   # col 6=vx_est, 7=vy_est
    gt   = d['pose_targets'].astype(np.float32)  # col 5=vx_gt,  6=vy_gt
    return gt[:, 5:7] - eskf[:, 6:8]

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
        legacy_files = []
    if EXCLUDED_FLIGHTS:
        flight_groups = {fid: chunks for fid, chunks in flight_groups.items()
                         if fid not in EXCLUDED_FLIGHTS}

print(f'Found {len(flight_groups)} new-format flight(s), {len(legacy_files)} legacy chunk(s)')

# ── Model ─────────────────────────────────────────────────────────────────────

INPUT_DIM = 23 if DATA_FOLDER == 'data_extended' else 15

class VelocityLSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):   # x: (1, T, 12)
        out, hidden = self.lstm(x, hidden)
        return self.fc(self.drop(out)), hidden   # (1, T, 2), hidden

# ── Training function ─────────────────────────────────────────────────────────

def train_model(train_seqs_norm, val_seqs_norm, label=''):
    model     = VelocityLSTM()
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30)

    epochs          = 500
    train_losses    = []
    val_losses      = []
    best_val_loss   = float('inf')
    best_epoch      = 0
    best_state_dict = None

    print(f'\nStarting training{" — " + label if label else ""}...', flush=True)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        perm = np.random.permutation(len(train_seqs_norm))
        for idx in perm:
            eskf_norm, delta_norm = train_seqs_norm[idx]
            hidden = None

            for i in range(0, len(eskf_norm), CHUNK_LEN):
                chunk_x = torch.tensor(eskf_norm[i:i + CHUNK_LEN]).unsqueeze(0)
                chunk_y = torch.tensor(delta_norm[i:i + CHUNK_LEN]).unsqueeze(0)

                if chunk_x.shape[1] < 2:
                    continue

                pred, hidden = model(chunk_x, hidden)
                hidden = (hidden[0].detach(),
                          torch.clamp(hidden[1].detach(), -100.0, 100.0))

                loss = loss_fn(pred, chunk_y)
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    hidden = None
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        train_losses.append(train_loss)

        model.eval()
        val_ep_losses = []
        with torch.no_grad():
            for eskf_norm, delta_norm, split in val_seqs_norm:
                hidden = None
                for i in range(0, split, CHUNK_LEN):
                    chunk_x = torch.tensor(eskf_norm[i:min(i + CHUNK_LEN, split)]).unsqueeze(0)
                    _, hidden = model(chunk_x, hidden)
                for i in range(split, len(eskf_norm), CHUNK_LEN):
                    chunk_x = torch.tensor(eskf_norm[i:i + CHUNK_LEN]).unsqueeze(0)
                    chunk_y = torch.tensor(delta_norm[i:i + CHUNK_LEN]).unsqueeze(0)
                    pred, hidden = model(chunk_x, hidden)
                    val_ep_losses.append(loss_fn(pred, chunk_y).item())

        v_loss = float(np.mean(val_ep_losses)) if val_ep_losses else float('inf')
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss   = v_loss
            best_epoch      = epoch + 1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        multiplicity = ' *** NEW VALIDATION BEST ***' if best_epoch == epoch + 1 else f' ---> No Improvement for {epoch - best_epoch + 2} epochs <---'

        print(f'Epoch {epoch+1}/{epochs}  Train MSE: {train_loss:.6f}  '
              f'Val MSE: {v_loss:.6f}  Best: {best_val_loss:.6f} @ epoch {best_epoch}' + multiplicity,
              flush=True)

        if best_val_loss <= VAL_MSE_TARGET:
            print(f'\nTarget val MSE {VAL_MSE_TARGET} reached at epoch {epoch+1}.')
            break

        if epoch + 1 - best_epoch >= EARLY_STOP_PATIENCE:
            print(f'\nEarly stop at epoch {epoch+1} — no improvement for {EARLY_STOP_PATIENCE} epochs.')
            break

    print(f'\nRestoring best model from epoch {best_epoch} (val MSE {best_val_loss:.6f})')
    model.load_state_dict(best_state_dict)
    return model, train_losses, val_losses, best_val_loss

def eval_rmse(model, val_seqs_norm, y_std, y_mean):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for eskf_norm, delta_norm, split in val_seqs_norm:
            hidden = None
            for i in range(0, split, CHUNK_LEN):
                chunk_x = torch.tensor(eskf_norm[i:min(i + CHUNK_LEN, split)]).unsqueeze(0)
                _, hidden = model(chunk_x, hidden)
            for i in range(split, len(eskf_norm), CHUNK_LEN):
                chunk_x = torch.tensor(eskf_norm[i:i + CHUNK_LEN]).unsqueeze(0)
                pred, hidden = model(chunk_x, hidden)
                all_preds.append(pred.squeeze(0).numpy())
                all_true.append(delta_norm[i:i + CHUNK_LEN])
    if not all_preds:
        return None
    y_pred_phys = np.concatenate(all_preds) * y_std + y_mean
    y_true_phys = np.concatenate(all_true)  * y_std + y_mean
    return [float(np.sqrt(np.mean((y_true_phys[:, i] - y_pred_phys[:, i])**2)))
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
    eskf_parts, dv_parts = [], []
    for p in chunk_paths:
        d = np.load(p, allow_pickle=True)
        eskf = _augment_eskf_inputs(d['eskf_inputs'].astype(np.float32), d)
        eskf_parts.append(eskf)
        dv_parts.append(_load_vxy(d))
    eskf_seq  = np.concatenate(eskf_parts, axis=0)
    delta_seq = np.concatenate(dv_parts,   axis=0)
    return eskf_seq, delta_seq

# ── Per-flight training ───────────────────────────────────────────────────────

if TRAIN_PER_FLIGHT:
    summary = []

    for fid, chunk_paths in sorted(flight_groups.items()):
        eskf_seq, delta_seq = load_flight(chunk_paths)
        N = len(eskf_seq)

        if N < MIN_FLIGHT_STEPS:
            print(f'\n── {fid}: {N} steps — too short, skipping ──')
            continue

        split = int(0.8 * N)
        print(f'\n{"─"*60}')
        print(f'Flight: {fid}  ({N} steps → {split} train / {N-split} val)')
        print(f'{"─"*60}')

        X_mean = eskf_seq[:split].mean(axis=0);  X_std = eskf_seq[:split].std(axis=0);   X_std[X_std < 1e-8] = 1.0
        y_mean = delta_seq[:split].mean(axis=0); y_std = delta_seq[:split].std(axis=0);  y_std[y_std < 1e-8] = 1.0

        print(f'Correction statistics (train):')
        for i, lbl in enumerate(LABELS):
            print(f'  {lbl}  mean={y_mean[i]:+.3f}  std={y_std[i]:.3f}')

        def norm_seq(eskf, delta):
            return ((eskf  - X_mean) / X_std).astype(np.float32), \
                   ((delta - y_mean) / y_std).astype(np.float32)

        en_full, dn_full = norm_seq(eskf_seq, delta_seq)
        train_seqs_norm = [(en_full[:split], dn_full[:split])]
        val_seqs_norm   = [(en_full, dn_full, split)]

        model, tl, vl, best_val = train_model(train_seqs_norm, val_seqs_norm, label=fid)
        plot_losses(tl, vl, f'ESKF Velocity LSTM — {fid}')

        rmse = eval_rmse(model, val_seqs_norm, y_std, y_mean)
        if rmse:
            print(f'\nPer-output validation RMSE — {fid}:')
            for lbl, r in zip(LABELS, rmse):
                print(f'  {lbl}: RMSE = {r:.4f} m/s')
            summary.append((fid, best_val, rmse))

        pt_path   = DATA_DIR / f'eskf_velocity_lstm_{fid}.pt'
        norm_path = DATA_DIR / f'eskf_velocity_lstm_{fid}_norm.npz'
        torch.save(model.state_dict(), pt_path)
        np.savez(norm_path, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
        print(f'Saved {pt_path.name}')

    if summary:
        print(f'\n{"═"*60}')
        print('PER-FLIGHT SUMMARY — eskf_velocity_lstm')
        print(f'{"═"*60}')
        hdr = f'{"Flight":<30}  {"Val MSE":>8}  ' + '  '.join(f'{l[:12]:>12}' for l in LABELS)
        print(hdr)
        for fid, bval, rmse in summary:
            row = f'{fid:<30}  {bval:>8.4f}  ' + '  '.join(f'{r:>12.4f}' for r in rmse)
            print(row)

        results = {fid: {'val_mse': bval, 'rmse': dict(zip(LABELS, rmse))}
                   for fid, bval, rmse in summary}
        out = DATA_DIR / 'results_eskf_velocity_lstm.json'
        out.write_text(json.dumps(results, indent=2))
        print(f'\nResults saved to {out.name}')

# ── Cross-flight training ─────────────────────────────────────────────────────

else:
    train_seqs = []
    val_seqs   = []

    if legacy_files:
        eskf_seq, delta_seq = load_flight(legacy_files)
        train_seqs.append((eskf_seq, delta_seq))
        print(f'  Legacy: {len(eskf_seq)} steps → train only')

    val_fids = []
    for fid, chunk_paths in flight_groups.items():
        is_mirror = '_mirror' in fid
        if is_mirror and not INCLUDE_MIRRORED:
            continue
        # VAL_FLIGHTS hold-out: exclude all mirror variants of val flights to prevent leakage
        if VAL_FLIGHTS and any(fid.startswith(v) and '_mirror' in fid for v in VAL_FLIGHTS):
            print(f'  {fid}: excluded (mirror of val flight)')
            continue
        eskf_seq, delta_seq = load_flight(chunk_paths)
        N = len(eskf_seq)
        if VAL_FLIGHTS and fid in VAL_FLIGHTS:
            # Cold-start val: split=0 → no warm-up, model sees this flight fresh
            val_seqs.append((eskf_seq, delta_seq, 0))
            val_fids.append(fid)
            print(f'  {fid}: {N} steps → VAL FLIGHT (cold-start, held out entirely)')
        elif is_mirror or N < MIN_FLIGHT_STEPS:
            train_seqs.append((eskf_seq, delta_seq))
            tag = 'mirror, train only' if is_mirror else 'train only (too short to split)'
            print(f'  {fid}: {N} steps → {tag}')
        else:
            split = int(0.8 * N) if not VAL_FLIGHTS else N
            train_seqs.append((eskf_seq[:split], delta_seq[:split]))
            if not VAL_FLIGHTS:
                val_seqs.append((eskf_seq, delta_seq, split))
                val_fids.append(fid)
            print(f'  {fid}: {N} steps → {"train" if VAL_FLIGHTS else f"{split} train, {N-split} val"}')

    all_train_eskf  = np.concatenate([s[0] for s in train_seqs], axis=0)
    all_train_delta = np.concatenate([s[1] for s in train_seqs], axis=0)
    X_mean = all_train_eskf.mean(axis=0);  X_std = all_train_eskf.std(axis=0);   X_std[X_std < 1e-8] = 1.0
    y_mean = all_train_delta.mean(axis=0); y_std = all_train_delta.std(axis=0);  y_std[y_std < 1e-8] = 1.0

    print(f'\nCorrection statistics (train):')
    for i, lbl in enumerate(LABELS):
        print(f'  {lbl}  mean={y_mean[i]:+.3f}  std={y_std[i]:.3f} m/s')
    print(f'  (Model must beat std to be useful)')

    def norm_seq(eskf, delta):
        return ((eskf  - X_mean) / X_std).astype(np.float32), \
               ((delta - y_mean) / y_std).astype(np.float32)

    train_seqs_norm = [norm_seq(e, d) for e, d in train_seqs]
    val_seqs_norm   = [(*norm_seq(e, d), s) for e, d, s in val_seqs]

    model, tl, vl, _ = train_model(train_seqs_norm, val_seqs_norm)
    plot_losses(tl, vl, 'ESKF Velocity LSTM — all flights')

    rmse = eval_rmse(model, val_seqs_norm, y_std, y_mean)
    if rmse:
        print('\nPer-output validation RMSE (physical units):')
        for lbl, r in zip(LABELS, rmse):
            print(f'  {lbl}: RMSE = {r:.4f} m/s')

    torch.save(model.state_dict(), DATA_DIR / 'eskf_velocity_lstm.pt')
    np.savez(DATA_DIR / 'eskf_velocity_lstm_norm.npz',
             X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    print(f'\nSaved model and normalisation to {DATA_DIR}')

    # ── Per-flight eval on val segments → JSON ────────────────────────────────
    cf_results = {}
    for fid, val_tuple in zip(val_fids, val_seqs_norm):
        rmse_f = eval_rmse(model, [val_tuple], y_std, y_mean)
        if rmse_f:
            fmse = float(np.mean([r**2 for r in rmse_f]))
            cf_results[fid] = {'val_mse': fmse, 'rmse': dict(zip(LABELS, rmse_f))}
            print(f'  {fid}  val_mse={fmse:.4f}  ' +
                  '  '.join(f'{l}={r:.4f}' for l, r in zip(LABELS, rmse_f)))
    if cf_results:
        out = DATA_DIR / 'results_eskf_velocity_lstm.json'
        out.write_text(json.dumps(cf_results, indent=2))
        print(f'Results saved to {out.name}')
