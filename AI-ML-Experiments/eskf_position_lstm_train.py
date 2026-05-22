"""
MARID ESKF Position LSTM Training
Trains an LSTM to predict position corrections (Δx, Δy) by processing each flight
as one continuous sequence with hidden state carried forward from takeoff.

- Input  (T, 12): ESKF state sequence [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
- Output (T, 2):  corrections at every timestep [Δx, Δy] = [x_true - x_eskf, y_true - y_eskf]

Why stateful vs windowed:
  The windowed LSTM starts mid-flight with no memory of accumulated drift.
  This LSTM resets hidden state only at flight start — by t=300s it has tracked
  every maneuver since takeoff and can accurately estimate total accumulated drift.

Modes (TRAIN_PER_FLIGHT):
  False — train one model on all flights combined (cross-flight generalisation).
  True  — train one model per flight independently. Each flight's Δx/Δy distribution
          dominates its own normalisation, so the LSTM sees a coherent drift
          trajectory without conflicting offsets from other flights.
          Models saved as eskf_position_lstm_<flight_id>.pt.

Training: Truncated BPTT (TBPTT) — process flight in CHUNK_LEN chunks,
  carry hidden state forward between chunks, detach gradients at chunk boundaries.

Val: warm up hidden state on train portion (first 80%), then evaluate on val portion
  (last 20%) — hidden state is correctly initialized, not cold-started.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── Dataset selection ─────────────────────────────────────────────────────────
# USE_EXTENDED=False → data/ (12-col base ESKF inputs, many old flights)
# USE_EXTENDED=True  → data_extended/ (23-col enriched: imu_acc, a_excess, yaw sources, airspeed)
#   Switch to True once ≥8 data_extended flights are available for a meaningful val set.
USE_EXTENDED = False
DATA_FOLDER  = 'data_extended' if USE_EXTENDED else 'data'
DATA_DIR     = Path(f"~/marid_ws/{DATA_FOLDER}").expanduser()

_G        = 9.81
_MIN_V    = 0.5
_PDOT_MAX = 1.5

CHUNK_LEN = 1000  # TBPTT chunk length (20s at 50 Hz) — position drift is ~8sec timescale

SELECTED_FLIGHTS    = []
EXCLUDED_FLIGHTS    = []
TRAIN_PER_FLIGHT    = False  # True → one model per flight; False → one model for all
EARLY_STOP_PATIENCE = 50    # position LSTM val is noisy — needs patience to find real improvement
VAL_MSE_TARGET      = 0.0
INCLUDE_MIRRORED    = True   # include logger-saved _mirror flights in train (never in val)
USE_AUTOREGRESSIVE  = False  # True → append Δ_{t-1} to LSTM input (input_dim+2);
                              # False → augmented ESKF state only
VAL_FLIGHTS         = [                            # hold these flights out as cold-start val;
    "flight_20260516_214622",  # existing benchmark: y-dominant drift
    "flight_20260516_221112",  # newer setup: x-dominant / cleaner y
    "flight_20260517_081702",  # newer setup: strong y drift
    "flight_20260517_083913",  # newer setup: x-dominant drift
    "flight_20260517_181254",  # newer setup: balanced shorter flight
    "flight_20260517_184327",  # newer setup: very y-dominant drift
    "flight_20260516_153217",  # transition/current-ish: very x-dominant drift
    "flight_20260514_150848",  # older large x/y drift, main-set stress without monster length
]                                                  # mirrors of all val flights excluded from train
# Long-horizon stress-test candidate:
#   flight_20260515_071720 — 94 min, >2 km final drift.
#   flight dominates the shorter validation flights.

_VAL_TIMESTAMPS = [v.replace('flight_', '') for v in VAL_FLIGHTS]  # for prefix-agnostic mirror exclusion

LABELS = ['Δx (m)', 'Δy (m)']
MIN_FLIGHT_STEPS = 2 * CHUNK_LEN

# ── Feature augmentation ──────────────────────────────────────────────────────

def _augment_eskf_inputs(eskf: np.ndarray, d) -> np.ndarray:
    """Augment ESKF inputs — mirrors eskf_velocity_lstm_train._augment_eskf_inputs.

    data/          → 15 cols: base(12) + thrust + ground_flag + psi_dot_aero
    data_extended/ → 23 cols: above(15) + imu_acc(3) + a_excess + delta_yaw_madgwick
                               + airspeed + delta_yaw_sun*sun_valid + sun_valid

    imu_acc preprocessing:
      ay clipped to ±15 m/s²  (Gazebo physics spikes)
      az gravity-subtracted   az_aero = az - g·cos(roll)·cos(pitch)
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

    imu_acc      = d['imu_acc'].astype(np.float32)[:N]
    imu_acc[:, 1] = np.clip(imu_acc[:, 1], -15.0, 15.0)
    a_excess = (np.linalg.norm(imu_acc, axis=1) - _G).astype(np.float32)
    imu_acc[:, 2] -= (_G * np.cos(roll) * np.cos(pitch)).astype(np.float32)
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

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_xy(d):
    """Return ground-truth [x, y], handling 7-D and 9-D legacy formats."""
    y = d['pose_targets'].astype(np.float32)
    return y[:, :2]

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

class PositionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(self.drop(out)), hidden   # (1, T, 2), hidden

_BASE_INPUT_DIM = 23 if USE_EXTENDED else 15
_LSTM_INPUT_DIM = _BASE_INPUT_DIM + 2 if USE_AUTOREGRESSIVE else _BASE_INPUT_DIM

# ── Training function ─────────────────────────────────────────────────────────

def train_model(train_seqs_norm, val_seqs_norm, label=''):
    model     = PositionLSTM(input_dim=_LSTM_INPUT_DIM)
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
            prev_delta = np.zeros(2, dtype=np.float32)  # (2,): d_{t-1} in norm space

            for i in range(0, len(eskf_norm), CHUNK_LEN):
                chunk_e = eskf_norm[i:i + CHUNK_LEN]  # (T, 12)
                chunk_d = delta_norm[i:i + CHUNK_LEN]  # (T, 2)

                if len(chunk_e) < 2:
                    continue

                if USE_AUTOREGRESSIVE:
                    ar = np.concatenate([prev_delta[np.newaxis], chunk_d[:-1]], axis=0)
                    chunk_x = torch.tensor(np.concatenate([chunk_e, ar], axis=-1)).unsqueeze(0)
                else:
                    chunk_x = torch.tensor(chunk_e).unsqueeze(0)
                chunk_y = torch.tensor(chunk_d).unsqueeze(0)

                pred, hidden = model(chunk_x, hidden)
                # Clamp cell state (c_t) to prevent float32 overflow on long flights.
                # h_t is bounded by tanh; c_t is unbounded and can grow to inf over
                # hundreds of chunks if forget gate ≈ 1, corrupting all subsequent gradients.
                hidden = (hidden[0].detach(),
                          torch.clamp(hidden[1].detach(), -100.0, 100.0))
                prev_delta = chunk_d[-1]  # GT carry-forward for next chunk (teacher forcing)

                loss = loss_fn(pred, chunk_y)
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    hidden = None  # reset hidden state on bad batch
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
                prev_delta = np.zeros(2, dtype=np.float32)
                # warmup: GT AR input, no loss (split=0 → loop skipped for cold-start)
                for i in range(0, split, CHUNK_LEN):
                    chunk_e = eskf_norm[i:min(i + CHUNK_LEN, split)]
                    chunk_d = delta_norm[i:min(i + CHUNK_LEN, split)]
                    if USE_AUTOREGRESSIVE:
                        ar = np.concatenate([prev_delta[np.newaxis], chunk_d[:-1]], axis=0)
                        chunk_x = torch.tensor(np.concatenate([chunk_e, ar], axis=-1)).unsqueeze(0)
                    else:
                        chunk_x = torch.tensor(chunk_e).unsqueeze(0)
                    _, hidden = model(chunk_x, hidden)
                    prev_delta = chunk_d[-1]
                # eval: AR uses model's own prediction (closed-loop, true inference condition)
                for i in range(split, len(eskf_norm), CHUNK_LEN):
                    chunk_e = eskf_norm[i:i + CHUNK_LEN]
                    chunk_d = delta_norm[i:i + CHUNK_LEN]
                    if USE_AUTOREGRESSIVE:
                        ar = np.concatenate([prev_delta[np.newaxis], chunk_d[:-1]], axis=0)
                        chunk_x = torch.tensor(np.concatenate([chunk_e, ar], axis=-1)).unsqueeze(0)
                    else:
                        chunk_x = torch.tensor(chunk_e).unsqueeze(0)
                    chunk_y = torch.tensor(chunk_d).unsqueeze(0)
                    pred, hidden = model(chunk_x, hidden)
                    prev_delta = pred.squeeze(0).numpy()[-1]  # closed-loop carry-forward
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
              f'Val MSE: {v_loss:.6f}  Best: {best_val_loss:.6f} @ epoch {best_epoch} ' + multiplicity,
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

def eval_validation_metrics(model, val_seqs_norm, y_std, y_mean):
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for eskf_norm, delta_norm, split in val_seqs_norm:
            hidden = None
            prev_delta = np.zeros(2, dtype=np.float32)
            for i in range(0, split, CHUNK_LEN):
                chunk_e = eskf_norm[i:min(i + CHUNK_LEN, split)]
                chunk_d = delta_norm[i:min(i + CHUNK_LEN, split)]
                if USE_AUTOREGRESSIVE:
                    ar = np.concatenate([prev_delta[np.newaxis], chunk_d[:-1]], axis=0)
                    chunk_x = torch.tensor(np.concatenate([chunk_e, ar], axis=-1)).unsqueeze(0)
                else:
                    chunk_x = torch.tensor(chunk_e).unsqueeze(0)
                _, hidden = model(chunk_x, hidden)
                prev_delta = chunk_d[-1]
            for i in range(split, len(eskf_norm), CHUNK_LEN):
                chunk_e = eskf_norm[i:i + CHUNK_LEN]
                chunk_d = delta_norm[i:i + CHUNK_LEN]
                if USE_AUTOREGRESSIVE:
                    ar = np.concatenate([prev_delta[np.newaxis], chunk_d[:-1]], axis=0)
                    chunk_x = torch.tensor(np.concatenate([chunk_e, ar], axis=-1)).unsqueeze(0)
                else:
                    chunk_x = torch.tensor(chunk_e).unsqueeze(0)
                pred, hidden = model(chunk_x, hidden)
                prev_delta = pred.squeeze(0).numpy()[-1]
                all_preds.append(pred.squeeze(0).numpy())
                all_true.append(chunk_d)
    if not all_preds:
        return None
    y_pred_phys = np.concatenate(all_preds) * y_std + y_mean
    y_true_phys = np.concatenate(all_true)  * y_std + y_mean
    model_rmse = np.sqrt(np.mean((y_true_phys - y_pred_phys) ** 2, axis=0))
    baseline_rmse = np.sqrt(np.mean(y_true_phys ** 2, axis=0))
    model_mse = float(np.mean((y_true_phys - y_pred_phys) ** 2))
    baseline_mse = float(np.mean(y_true_phys ** 2))
    improvement = np.zeros_like(model_rmse)
    valid = baseline_rmse > 1e-8
    improvement[valid] = 100.0 * (baseline_rmse[valid] - model_rmse[valid]) / baseline_rmse[valid]
    return {
        'model_rmse': [float(v) for v in model_rmse],
        'baseline_rmse': [float(v) for v in baseline_rmse],
        'model_mse': model_mse,
        'baseline_mse': baseline_mse,
        'improvement_pct': [float(v) for v in improvement],
    }

def eval_rmse(model, val_seqs_norm, y_std, y_mean):
    metrics = eval_validation_metrics(model, val_seqs_norm, y_std, y_mean)
    return None if metrics is None else metrics['model_rmse']

def print_validation_metrics(metrics, title='Per-output validation RMSE (physical units)'):
    if not metrics:
        return
    print(f'\n{title}:')
    for lbl, model_r, base_r, imp in zip(
            LABELS,
            metrics['model_rmse'],
            metrics['baseline_rmse'],
            metrics['improvement_pct']):
        print(
            f'  {lbl}: model={model_r:.4f} m  '
            f'baseline={base_r:.4f} m  improvement={imp:+.1f}%'
        )
    mse_imp = 0.0
    if metrics['baseline_mse'] > 1e-8:
        mse_imp = 100.0 * (metrics['baseline_mse'] - metrics['model_mse']) / metrics['baseline_mse']
    print(
        f'  mean MSE: model={metrics["model_mse"]:.4f}  '
        f'baseline={metrics["baseline_mse"]:.4f}  improvement={mse_imp:+.1f}%'
    )

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
    eskf_parts, gt_parts = [], []
    for p in chunk_paths:
        d = np.load(p, allow_pickle=True)
        eskf = _augment_eskf_inputs(d['eskf_inputs'].astype(np.float32), d)
        eskf_parts.append(eskf)
        gt_parts.append(_load_xy(d))
    eskf_seq  = np.concatenate(eskf_parts, axis=0)
    gt_xy     = np.concatenate(gt_parts,   axis=0)
    delta_seq = gt_xy - eskf_seq[:, :2]
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
            print(f'  {lbl}  mean={y_mean[i]:+.2f}  std={y_std[i]:.2f}')

        def norm_seq(eskf, delta):
            return ((eskf  - X_mean) / X_std).astype(np.float32), \
                   ((delta - y_mean) / y_std).astype(np.float32)

        en_full, dn_full = norm_seq(eskf_seq, delta_seq)
        train_seqs_norm = [(en_full[:split], dn_full[:split])]
        val_seqs_norm   = [(en_full, dn_full, split)]

        model, tl, vl, best_val = train_model(train_seqs_norm, val_seqs_norm, label=fid)
        plot_losses(tl, vl, f'ESKF Position LSTM — {fid}')

        metrics = eval_validation_metrics(model, val_seqs_norm, y_std, y_mean)
        if metrics:
            print_validation_metrics(metrics, title=f'Validation RMSE — {fid}')
            summary.append((fid, best_val, metrics['model_rmse']))

        pt_path   = DATA_DIR / f'eskf_position_lstm_{fid}.pt'
        norm_path = DATA_DIR / f'eskf_position_lstm_{fid}_norm.npz'
        torch.save(model.state_dict(), pt_path)
        np.savez(norm_path, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std,
                 use_autoregressive=USE_AUTOREGRESSIVE)
        print(f'Saved {pt_path.name}')

    if summary:
        print(f'\n{"═"*60}')
        print('PER-FLIGHT SUMMARY — eskf_position_lstm')
        print(f'{"═"*60}')
        hdr = f'{"Flight":<30}  {"Val MSE":>8}  ' + '  '.join(f'{l[:8]:>10}' for l in LABELS)
        print(hdr)
        for fid, bval, rmse in summary:
            row = f'{fid:<30}  {bval:>8.4f}  ' + '  '.join(f'{r:>10.3f}' for r in rmse)
            print(row)

        results = {fid: {'val_mse': bval, 'rmse': dict(zip(LABELS, rmse))}
                   for fid, bval, rmse in summary}
        out = DATA_DIR / 'results_eskf_position_lstm.json'
        out.write_text(json.dumps(results, indent=2))
        print(f'\nResults saved to {out.name}')

# ── Cross-flight training (original mode) ─────────────────────────────────────

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
        # VAL_FLIGHT hold-out: exclude all mirror variants of the val flight to prevent leakage
        if VAL_FLIGHTS and any(ts in fid and '_mirror' in fid for ts in _VAL_TIMESTAMPS):
            print(f'  {fid}: excluded (mirror of val flight)')
            continue
        eskf_seq, delta_seq = load_flight(chunk_paths)
        N = len(eskf_seq)
        if VAL_FLIGHTS and fid in VAL_FLIGHTS:
            # Val: warm up hidden state on first CHUNK_LEN steps, evaluate on the rest.
            # Pure cold-start (split=0) causes extreme epoch-to-epoch variance because
            # the LSTM has no drift context for the first few minutes of the flight.
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

    all_train_eskf = np.concatenate([s[0] for s in train_seqs], axis=0)
    X_mean = all_train_eskf.mean(axis=0); X_std = all_train_eskf.std(axis=0); X_std[X_std < 1e-8] = 1.0

    # No per-flight zero-centering: val flights start at Δ=0 and drift monotonically,
    # so train targets must also be in the same absolute-drift space. Zero-centering
    # would teach the LSTM centered-oscillation targets which mis-match val inference.
    all_train_delta = np.concatenate([s[1] for s in train_seqs], axis=0)
    y_std  = all_train_delta.std(axis=0); y_std[y_std < 1e-8] = 1.0
    y_mean = np.zeros(2, dtype=np.float32)
    train_delta_means = [np.zeros(2, dtype=np.float32) for _ in train_seqs]

    print(f'\nCorrection statistics (train, absolute drift):')
    for i, lbl in enumerate(LABELS):
        print(f'  {lbl}  global std={y_std[i]:.2f} m  mean={all_train_delta.mean(axis=0)[i]:.2f} m')
    print(f'  (Model must beat std to be useful)')

    def norm_seq(eskf, delta, delta_mean):
        return ((eskf  - X_mean) / X_std).astype(np.float32), \
               ((delta - delta_mean) / y_std).astype(np.float32)

    train_seqs_norm = [norm_seq(e, d, m) for (e, d), m in zip(train_seqs, train_delta_means)]
    # val: zero-centre using zeros (flight starts at Δ=0 so no mean to subtract)
    val_seqs_norm = []
    for e, d, s in val_seqs:
        delta_mean = np.zeros(2, dtype=np.float32)

        en, dn = norm_seq(e, d, delta_mean)
        val_seqs_norm.append((en, dn, s))

    model, tl, vl, _ = train_model(train_seqs_norm, val_seqs_norm)
    plot_losses(tl, vl, 'ESKF Position LSTM — all flights')

    metrics = eval_validation_metrics(model, val_seqs_norm, y_std, y_mean)
    if metrics:
        print_validation_metrics(metrics)

    torch.save(model.state_dict(), DATA_DIR / 'eskf_position_lstm.pt')
    np.savez(DATA_DIR / 'eskf_position_lstm_norm.npz',
             X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std,
             use_autoregressive=USE_AUTOREGRESSIVE)
    print(f'\nSaved model and normalization to {DATA_DIR}')

    # ── Per-flight eval on val segments → JSON for compare_results.py ────────
    cf_results = {}
    for fid, val_tuple in zip(val_fids, val_seqs_norm):
        metrics_f = eval_validation_metrics(model, [val_tuple], y_std, y_mean)
        if metrics_f:
            cf_results[fid] = {
                'val_mse': metrics_f['model_mse'],
                'baseline_mse': metrics_f['baseline_mse'],
                'rmse': dict(zip(LABELS, metrics_f['model_rmse'])),
                'baseline_rmse': dict(zip(LABELS, metrics_f['baseline_rmse'])),
                'improvement_pct': dict(zip(LABELS, metrics_f['improvement_pct'])),
            }
            print(f'  {fid}  val_mse={metrics_f["model_mse"]:.4f}  '
                  f'baseline_mse={metrics_f["baseline_mse"]:.4f}  ' +
                  '  '.join(
                      f'{l}: model={r:.3f}, base={b:.3f}, imp={imp:+.1f}%'
                      for l, r, b, imp in zip(
                          LABELS,
                          metrics_f['model_rmse'],
                          metrics_f['baseline_rmse'],
                          metrics_f['improvement_pct'])
                  ))
    if cf_results:
        out = DATA_DIR / 'results_eskf_position_lstm.json'
        out.write_text(json.dumps(cf_results, indent=2))
        print(f'Results saved to {out.name}')
