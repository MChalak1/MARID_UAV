"""
MARID ESKF Attitude LSTM Training
Stateful LSTM predicting attitude and velocity corrections from the full ESKF
state sequence. Compared against the feedforward version (eskf_attitude_ff_train.py)
to determine whether temporal memory improves attitude/velocity estimation.

- Input  (T, 12): ESKF state sequence [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
- Output (T, 4):  [roll, pitch, vx, vy] — absolute ground-truth values
    roll, pitch  : absolute ground-truth attitude (yaw excluded — history-dependent drift)
    vx, vy       : absolute ground-truth velocity

Position (Δx, Δy) is handled by eskf_position_lstm_train.py — kept separate because
unbounded position drift and bounded attitude/velocity have incompatible training
dynamics in a shared network.

Modes (TRAIN_PER_FLIGHT):
  False — train one model on all flights combined (cross-flight generalisation).
          Normalisation is fit on all training data pooled.
  True  — train one model per flight independently (per-flight specialisation).
          Models saved as eskf_attitude_lstm_<flight_id>.pt.

Training: Truncated BPTT (CHUNK_LEN=200 steps, 4 s at 50 Hz).
Val:      Warm up hidden state on train portion (first 80%), then evaluate on
          the remaining 20% — hidden state is correctly initialised, not cold.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

DATA_DIR  = Path("~/marid_ws/data").expanduser()
CHUNK_LEN = 200   # TBPTT chunk length (4 s at 50 Hz)

SELECTED_FLIGHTS    = []
EXCLUDED_FLIGHTS    = []
TRAIN_PER_FLIGHT    = False  # True → one model per flight; False → one model for all
EARLY_STOP_PATIENCE = 150
VAL_MSE_TARGET      = 0.05
INCLUDE_MIRRORED = True  # Whether to include mirrored flights in training (if False, all mirrors are excluded).
VAL_FLIGHT = "flight_20260508_213617"

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_targets(d):
    """Return (T, 4): [roll, pitch, vx, vy] — yaw and position excluded.
    7-D current format [x, y, roll, pitch, yaw, vx, vy] → cols [2, 3, 5, 6].
    9-D legacy format  [x, y, z, roll, pitch, yaw, vx, vy, vz] → cols [3, 4, 6, 7].
    """
    y = d['pose_targets'].astype(np.float32)
    if y.shape[1] == 9:
        return y[:, [3, 4, 6, 7]]
    return y[:, [2, 3, 5, 6]]

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

LABELS = ['roll (rad)', 'pitch (rad)', 'vx (m/s)', 'vy (m/s)']
MIN_FLIGHT_STEPS = 2 * CHUNK_LEN

# ── Model definition ──────────────────────────────────────────────────────────

class AttitudeLSTM(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=256, num_layers=2, output_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):   # x: (1, T, 12)
        out, hidden = self.lstm(x, hidden)
        return self.fc(self.drop(out)), hidden   # (1, T, 4), hidden

# ── Training function (shared by both modes) ──────────────────────────────────

def train_model(train_seqs_norm, val_seqs_norm, label=''):
    model     = AttitudeLSTM()
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20)

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
            eskf_norm, tgt_norm = train_seqs_norm[idx]
            hidden = None

            for i in range(0, len(eskf_norm), CHUNK_LEN):
                chunk_x = torch.tensor(eskf_norm[i:i + CHUNK_LEN]).unsqueeze(0)
                chunk_y = torch.tensor(tgt_norm[i:i + CHUNK_LEN]).unsqueeze(0)

                if chunk_x.shape[1] < 2:
                    continue

                pred, hidden = model(chunk_x, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())

                loss = loss_fn(pred, chunk_y)
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
            for eskf_norm, tgt_norm, split in val_seqs_norm:
                hidden = None

                for i in range(0, split, CHUNK_LEN):
                    chunk_x = torch.tensor(eskf_norm[i:min(i + CHUNK_LEN, split)]).unsqueeze(0)
                    _, hidden = model(chunk_x, hidden)

                for i in range(split, len(eskf_norm), CHUNK_LEN):
                    chunk_x = torch.tensor(eskf_norm[i:i + CHUNK_LEN]).unsqueeze(0)
                    chunk_y = torch.tensor(tgt_norm[i:i + CHUNK_LEN]).unsqueeze(0)
                    pred, hidden = model(chunk_x, hidden)
                    val_ep_losses.append(loss_fn(pred, chunk_y).item())

        v_loss = float(np.mean(val_ep_losses)) if val_ep_losses else float('inf')
        val_losses.append(v_loss)
        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss   = v_loss
            best_epoch      = epoch + 1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        print(f'Epoch {epoch+1}/{epochs}  Train MSE: {train_loss:.6f}  '
              f'Val MSE: {v_loss:.6f}  Best: {best_val_loss:.6f} @ epoch {best_epoch}',
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
    """Return per-output RMSE in physical units."""
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for eskf_norm, tgt_norm, split in val_seqs_norm:
            hidden = None
            for i in range(0, split, CHUNK_LEN):
                chunk_x = torch.tensor(eskf_norm[i:min(i + CHUNK_LEN, split)]).unsqueeze(0)
                _, hidden = model(chunk_x, hidden)
            for i in range(split, len(eskf_norm), CHUNK_LEN):
                chunk_x = torch.tensor(eskf_norm[i:i + CHUNK_LEN]).unsqueeze(0)
                pred, hidden = model(chunk_x, hidden)
                all_preds.append(pred.squeeze(0).numpy())
                all_true.append(tgt_norm[i:i + CHUNK_LEN])
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

# ── Build flight sequences ─────────────────────────────────────────────────────

def load_flight(fid, chunk_paths):
    eskf_parts, tgt_parts = [], []
    for p in chunk_paths:
        d = np.load(p, allow_pickle=True)
        eskf_parts.append(d['eskf_inputs'].astype(np.float32))
        tgt_parts.append(_load_targets(d))
    eskf_seq = np.concatenate(eskf_parts, axis=0)
    tgt_seq  = np.concatenate(tgt_parts,  axis=0)
    return eskf_seq, tgt_seq

# ── Per-flight training ───────────────────────────────────────────────────────

if TRAIN_PER_FLIGHT:
    summary = []   # (fid, best_val_loss, rmse_list)

    for fid, chunk_paths in sorted(flight_groups.items()):
        eskf_seq, tgt_seq = load_flight(fid, chunk_paths)
        N = len(eskf_seq)

        if N < MIN_FLIGHT_STEPS:
            print(f'\n── {fid}: {N} steps — too short, skipping ──')
            continue

        split = int(0.8 * N)
        print(f'\n{"─"*60}')
        print(f'Flight: {fid}  ({N} steps → {split} train / {N-split} val)')
        print(f'{"─"*60}')

        # Normalise on this flight's train portion only
        X_mean = eskf_seq[:split].mean(axis=0)
        X_std  = eskf_seq[:split].std(axis=0);  X_std[X_std < 1e-8] = 1.0
        y_mean = tgt_seq[:split].mean(axis=0)
        y_std  = tgt_seq[:split].std(axis=0);   y_std[y_std < 1e-8] = 1.0

        print('Target statistics (train):')
        for i, lbl in enumerate(LABELS):
            print(f'  {lbl:14s}  mean={y_mean[i]:+.3f}  std={y_std[i]:.3f}')

        def norm_seq(eskf, tgt):
            return ((eskf - X_mean) / X_std).astype(np.float32), \
                   ((tgt  - y_mean) / y_std).astype(np.float32)

        en_full, tn_full = norm_seq(eskf_seq, tgt_seq)
        train_seqs_norm = [(en_full[:split], tn_full[:split])]
        val_seqs_norm   = [(en_full, tn_full, split)]

        model, tl, vl, best_val = train_model(
            train_seqs_norm, val_seqs_norm, label=fid)

        plot_losses(tl, vl, f'Attitude LSTM — {fid}')

        rmse = eval_rmse(model, val_seqs_norm, y_std, y_mean)
        if rmse:
            print(f'\nPer-output validation RMSE — {fid}:')
            for lbl, r in zip(LABELS, rmse):
                print(f'  {lbl:14s}: RMSE = {r:.4f}')
            summary.append((fid, best_val, rmse))

        # Save per-flight model
        pt_path   = DATA_DIR / f'eskf_attitude_lstm_{fid}.pt'
        norm_path = DATA_DIR / f'eskf_attitude_lstm_{fid}_norm.npz'
        torch.save(model.state_dict(), pt_path)
        np.savez(norm_path, X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
        print(f'Saved {pt_path.name}')

    # ── Summary across all flights ───────────────────────────────��─────────────
    if summary:
        print(f'\n{"═"*60}')
        print('PER-FLIGHT SUMMARY — eskf_attitude_lstm (LSTM)')
        print(f'{"═"*60}')
        hdr = f'{"Flight":<30}  {"Val MSE":>8}  ' + '  '.join(f'{l[:6]:>8}' for l in LABELS)
        print(hdr)
        for fid, bval, rmse in summary:
            row = f'{fid:<30}  {bval:>8.4f}  ' + '  '.join(f'{r:>8.3f}' for r in rmse)
            print(row)

        results = {fid: {'val_mse': bval, 'rmse': dict(zip(LABELS, rmse))}
                   for fid, bval, rmse in summary}
        out = DATA_DIR / 'results_eskf_attitude_lstm.json'
        out.write_text(json.dumps(results, indent=2))
        print(f'\nResults saved to {out.name}')

# ── Cross-flight training (original mode) ─────────────────────────────────────

# ── Cross-flight training (updated with optional cold-flight validation) ─────

else:
    all_train_seqs = []
    all_val_seqs   = []

    if legacy_files:
        eskf_parts, tgt_parts = [], []

        for p in legacy_files:
            d = np.load(p, allow_pickle=True)
            eskf_parts.append(d['eskf_inputs'].astype(np.float32))
            tgt_parts.append(_load_targets(d))

        eskf_seq = np.concatenate(eskf_parts, axis=0)
        tgt_seq  = np.concatenate(tgt_parts,  axis=0)

        all_train_seqs.append((eskf_seq, tgt_seq))
        print(f'  Legacy: {len(eskf_seq)} steps → train only')

    val_fids = []

    for fid, chunk_paths in flight_groups.items():

        is_mirror = "_mirror" in fid

        if is_mirror and not INCLUDE_MIRRORED:
            continue

        # Exclude mirrors of held-out validation flight
        if VAL_FLIGHT and fid.startswith(VAL_FLIGHT) and "_mirror" in fid:
            print(f'  {fid}: excluded (mirror of val flight)')
            continue

        eskf_seq, tgt_seq = load_flight(fid, chunk_paths)
        N = len(eskf_seq)

        # True held-out cold validation flight
        if VAL_FLIGHT and fid == VAL_FLIGHT:
            all_val_seqs.append((eskf_seq, tgt_seq, 0))   # split=0 → cold start
            val_fids.append(fid)

            print(f'  {fid}: {N} steps → VAL FLIGHT (cold-start, held out entirely)')
            continue

        # Mirror flights → train only
        if is_mirror:
            all_train_seqs.append((eskf_seq, tgt_seq))
            print(f'  {fid}: {N} steps → mirror, train only')
            continue

        # Too short → train only
        if N < MIN_FLIGHT_STEPS:
            all_train_seqs.append((eskf_seq, tgt_seq))
            print(f'  {fid}: {N} steps → train only (too short to split)')
            continue

        # NORMAL MODE (no VAL_FLIGHT)
        if not VAL_FLIGHT:

            split = int(0.8 * N)

            all_train_seqs.append((eskf_seq[:split], tgt_seq[:split]))
            all_val_seqs.append((eskf_seq, tgt_seq, split))

            val_fids.append(fid)

            print(f'  {fid}: {N} steps → {split} train, {N-split} val')

        # HOLD-OUT MODE (VAL_FLIGHT set)
        else:

            all_train_seqs.append((eskf_seq, tgt_seq))

            print(f'  {fid}: {N} steps → train')

    # ── Normalization ─────────────────────────────────────────────────────────

    all_train_eskf = np.concatenate([s[0] for s in all_train_seqs], axis=0)
    all_train_tgt  = np.concatenate([s[1] for s in all_train_seqs], axis=0)

    X_mean = all_train_eskf.mean(axis=0)
    X_std  = all_train_eskf.std(axis=0)
    X_std[X_std < 1e-8] = 1.0

    y_mean = all_train_tgt.mean(axis=0)
    y_std  = all_train_tgt.std(axis=0)
    y_std[y_std < 1e-8] = 1.0

    print('\nTarget statistics (train):')

    for i, lbl in enumerate(LABELS):
        print(f'  {lbl:14s}  mean={y_mean[i]:+.3f}  std={y_std[i]:.3f}')

    print('  (Model must beat std to be useful)')

    def norm_seq(eskf, tgt):
        return ((eskf - X_mean) / X_std).astype(np.float32), \
               ((tgt  - y_mean) / y_std).astype(np.float32)

    train_seqs_norm = [norm_seq(e, t) for e, t in all_train_seqs]
    val_seqs_norm   = [(*norm_seq(e, t), s) for e, t, s in all_val_seqs]

    # ── Train ────────────────────────────────────────────────────────────────

    model, tl, vl, _ = train_model(train_seqs_norm, val_seqs_norm)

    plot_losses(
        tl,
        vl,
        'ESKF Unified Stateful LSTM — all flights'
        if not VAL_FLIGHT
        else 'ESKF Unified Stateful LSTM — cold-flight validation'
    )

    # ── Global RMSE ──────────────────────────────────────────────────────────

    rmse = eval_rmse(model, val_seqs_norm, y_std, y_mean)

    if rmse:
        print('\nPer-output validation RMSE (physical units):')

        for lbl, r in zip(LABELS, rmse):
            print(f'  {lbl:14s}: RMSE = {r:.4f}')

    # ── Save model ───────────────────────────────────────────────────────────

    torch.save(model.state_dict(), DATA_DIR / 'eskf_attitude_lstm.pt')

    np.savez(
        DATA_DIR / 'eskf_attitude_lstm_norm.npz',
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std
    )

    print(f'\nSaved model and normalization to {DATA_DIR}')

    # ── Per-flight eval → JSON ──────────────────────────────────────────────

    cf_results = {}

    for fid, val_tuple in zip(val_fids, val_seqs_norm):

        rmse_f = eval_rmse(model, [val_tuple], y_std, y_mean)

        if rmse_f:

            fmse = float(np.mean([r**2 for r in rmse_f]))

            cf_results[fid] = {
                'val_mse': fmse,
                'rmse': dict(zip(LABELS, rmse_f))
            }

            print(
                f'  {fid}  val_mse={fmse:.4f}  ' +
                '  '.join(f'{l}={r:.3f}' for l, r in zip(LABELS, rmse_f))
            )

    if cf_results:
        out = DATA_DIR / 'results_eskf_attitude_lstm.json'
        out.write_text(json.dumps(cf_results, indent=2))

        print(f'Results saved to {out.name}')
