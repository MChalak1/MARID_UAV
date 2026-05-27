"""
cascade_eval.py — Cascade correction evaluation on held-out val flights.

Tests three configurations on the velocity-LSTM val flights (cold-start,
never seen during velocity LSTM training):

  Config A  ESKF only        — raw ESKF velocity dead-reckoning from GT pos at t=0
  Config B  + Yaw FF         — Δψ → rotate world-frame velocity → integrate
  Config C  + Vel LSTM(raw)  — LSTM Δvx/Δvy applied to RAW ESKF velocity (independent of B)

Why no Config D (cascade = Yaw FF + LSTM)?
  The velocity LSTM was trained on raw ESKF errors — which are dominated by uncorrected
  yaw error. Applying the LSTM on top of yaw-corrected velocity feeds it contradicting
  pose information, causing double-counting and degraded performance. Removed entirely.

Dead-reckoning reference:
  gt_pos is published at ~10-17 Hz (repeated 3-9× at 50 Hz logging rate).
  Comparing ∫eskf_vel against raw gt_pos inflates error ~4×.
  Correct reference: pos_gt_dr = ∫gt_vel from the same starting point.
  All position errors are measured against pos_gt_dr.

Metrics per config per flight:
  - Yaw RMSE (deg)
  - Velocity RMSE: vx, vy, combined (m/s)
  - Dead-reckoning position RMSE at 30 s, 60 s, 120 s, 300 s, end-of-flight (m)
  - Position drift rate (m/min, linear fit on Euclidean error vs time)
  - ESKF sensor-fusion position RMSE vs Gazebo gt_pos (separate from DR errors)

Output: console table + cascade_eval_results.json + cascade_eval_plots.png in data_extended/
"""

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')       # headless — comment out if running in Jupyter
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR  = Path('~/marid_ws/data_extended').expanduser()
OUT_JSON  = DATA_DIR / 'cascade_eval_results.json'
OUT_PLOT  = DATA_DIR / 'cascade_eval_plots.png'

DT        = 1.0 / 50.0   # 50 Hz logging rate [s]
CHUNK_LEN = 300           # LSTM inference chunk (must match training CHUNK_LEN)

# Val flights held out from velocity LSTM training (cold-start, no data leakage)
VAL_FLIGHTS = [
    'flight_20260521_211150',   #  8.9 min, 2D spread 1041×1304 m, yaw err ~4.9°
    'flight_20260522_062211',   # 14.6 min, 2D spread 2546×2365 m, yaw err ~4.8°
    'flight_20260523_090404',   # 29.1 min, 2D spread 4136×3180 m, yaw err ~3.7° (longest)
]

# Snapshot times for position error table [s]
SNAPSHOT_TIMES = [30, 60, 120, 300]

# ── Helpers ───────────────────────────────────────────────────────────────────

_G        = 9.81
_MIN_V    = 3.0
_PDOT_MAX = 2.0


def _wrap(a):
    return ((a + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def _augment_eskf_inputs(eskf: np.ndarray, d: dict) -> np.ndarray:
    """23-col augmented input — identical to both training scripts."""
    N           = len(eskf)
    thrust      = d['thrust'].astype(np.float32).ravel()[:N] if 'thrust' in d else np.zeros(N, np.float32)
    z           = eskf[:, 2]
    roll        = eskf[:, 3]
    pitch       = eskf[:, 4]
    yaw_eskf    = eskf[:, 5]
    vx, vy      = eskf[:, 6], eskf[:, 7]
    ground_flag = (z < 1.0).astype(np.float32)

    if 'airspeed' in d:
        V_src = np.clip(np.abs(d['airspeed'].astype(np.float32).ravel()[:N]), _MIN_V, None)
    else:
        V_src = np.clip(np.sqrt(vx**2 + vy**2), _MIN_V, None)

    psi_dot = np.clip((_G / V_src) * np.tan(roll), -_PDOT_MAX, _PDOT_MAX).astype(np.float32)
    psi_dot *= (1.0 - ground_flag)

    base = np.concatenate([eskf, thrust[:, None], ground_flag[:, None], psi_dot[:, None]], axis=1)

    if 'imu_acc' not in d:
        return base   # 15 cols (legacy)

    imu_acc          = d['imu_acc'].astype(np.float32)[:N]
    imu_acc[:, 1]    = np.clip(imu_acc[:, 1], -15.0, 15.0)
    a_excess         = (np.linalg.norm(imu_acc, axis=1) - _G).astype(np.float32)
    imu_acc[:, 2]   -= (_G * np.cos(roll) * np.cos(pitch)).astype(np.float32)
    yaw_madgwick     = d['yaw_madgwick'].astype(np.float32).ravel()[:N]
    airspeed         = d['airspeed'].astype(np.float32).ravel()[:N]
    sun_yaw          = d['sun_yaw'].astype(np.float32).ravel()[:N]
    sun_valid        = d['sun_valid'].astype(np.float32).ravel()[:N]

    delta_madgwick   = _wrap(yaw_madgwick - yaw_eskf)
    delta_sun        = _wrap(sun_yaw      - yaw_eskf) * sun_valid

    return np.concatenate([base,
                           imu_acc,
                           a_excess[:, None],
                           delta_madgwick[:, None],
                           airspeed[:, None],
                           delta_sun[:, None],
                           sun_valid[:, None]], axis=1)   # 23 cols


def load_flight(fid: str):
    """Load and concatenate all chunks for a flight.

    Returns
    -------
    feat   : (T, 23) float32   augmented ESKF input
    eskf   : (T, 12) float32   raw ESKF state
    gt_pos : (T, 2)  float32   [x_gt, y_gt] m  (subsampled ~10-17 Hz; use only as absolute ref)
    gt_yaw : (T,)    float32   yaw_gt rad
    gt_vel : (T, 2)  float32   [vx_gt, vy_gt] m/s  (world frame, 50 Hz — use for DR reference)
    """
    chunks = sorted(DATA_DIR.glob(f'marid_eskf_gt_{fid}_chunk*.npz'))
    if not chunks:
        raise FileNotFoundError(f'No chunks found for {fid} in {DATA_DIR}')

    feat_parts, eskf_parts, gt_pos_parts, gt_yaw_parts, gt_vel_parts = [], [], [], [], []
    for p in chunks:
        d    = np.load(p, allow_pickle=True)
        eskf = d['eskf_inputs'].astype(np.float32)
        gt   = d['pose_targets'].astype(np.float32)   # (N, 7): [x,y,roll,pitch,yaw,vx,vy]

        feat_parts.append(_augment_eskf_inputs(eskf, d))
        eskf_parts.append(eskf)
        gt_pos_parts.append(gt[:, 0:2])
        gt_yaw_parts.append(gt[:, 4])
        gt_vel_parts.append(gt[:, 5:7])

    return (np.concatenate(feat_parts),
            np.concatenate(eskf_parts),
            np.concatenate(gt_pos_parts),
            np.concatenate(gt_yaw_parts),
            np.concatenate(gt_vel_parts))


# ── Model definitions (must match training scripts exactly) ───────────────────

def make_attitude_ff(input_dim: int = 23) -> nn.Sequential:
    """Matches make_model() in eskf_attitude_ff_train.py exactly (bare Sequential)."""
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256,       256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256,         1),
    )


class VelocityLSTM(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.3)
        self.drop = nn.Dropout(0.4)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(self.drop(out)), hidden


# ── Load models ───────────────────────────────────────────────────────────────

print('Loading models...')

# Attitude FF
_ff_norm  = np.load(DATA_DIR / 'eskf_attitude_ff_norm.npz')
ff_X_mean = _ff_norm['X_mean'].astype(np.float32)
ff_X_std  = _ff_norm['X_std'].astype(np.float32)
ff_y_mean = float(_ff_norm['y_mean'])
ff_y_std  = float(_ff_norm['y_std'])

att_model = make_attitude_ff()
att_model.load_state_dict(torch.load(DATA_DIR / 'eskf_attitude_ff.pt', map_location='cpu', weights_only=True))
att_model.eval()
print(f'  Attitude FF loaded  (y_mean={ff_y_mean:.4f} rad, y_std={ff_y_std:.4f} rad)')

# Velocity LSTM
_vel_norm  = np.load(DATA_DIR / 'eskf_velocity_lstm_norm.npz')
vel_X_mean = _vel_norm['X_mean'].astype(np.float32)
vel_X_std  = _vel_norm['X_std'].astype(np.float32)
vel_y_mean = _vel_norm['y_mean'].astype(np.float32)   # (2,)
vel_y_std  = _vel_norm['y_std'].astype(np.float32)    # (2,)

vel_model = VelocityLSTM()
vel_model.load_state_dict(torch.load(DATA_DIR / 'eskf_velocity_lstm.pt', map_location='cpu', weights_only=True))
vel_model.eval()
print(f'  Velocity LSTM loaded (y_mean={vel_y_mean}, y_std={vel_y_std})')


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_yaw_correction(feat: np.ndarray) -> np.ndarray:
    """Δψ for every timestep [rad], pointwise (no hidden state)."""
    X = torch.tensor((feat - ff_X_mean) / ff_X_std)
    with torch.no_grad():
        y_norm = att_model(X).numpy().ravel()
    return (y_norm * ff_y_std + ff_y_mean).astype(np.float32)


def predict_velocity_correction(feat: np.ndarray) -> np.ndarray:
    """[Δvx, Δvy] for every timestep [m/s]. LSTM is cold-started (h=None).

    Processes the full flight in CHUNK_LEN=300 chunks, carrying hidden state
    forward — this is the real deployment scenario (no warm-up from GT data).
    """
    T     = len(feat)
    delta = np.zeros((T, 2), dtype=np.float32)
    X_norm = ((feat - vel_X_mean) / vel_X_std).astype(np.float32)

    hidden = None
    with torch.no_grad():
        for i in range(0, T, CHUNK_LEN):
            chunk = torch.tensor(X_norm[i:i + CHUNK_LEN]).unsqueeze(0)  # (1, L, 23)
            pred, hidden = vel_model(chunk, hidden)
            # Clamp cell state to prevent float32 overflow on long flights
            hidden = (hidden[0].detach(),
                      torch.clamp(hidden[1].detach(), -100.0, 100.0))
            raw = pred.squeeze(0).numpy()   # (L, 2) normalised
            delta[i:i + len(raw)] = raw * vel_y_std + vel_y_mean

    return delta


def rotate_velocity(vx: np.ndarray, vy: np.ndarray,
                    dpsi: np.ndarray) -> tuple:
    """Rotate world-frame velocity by Δψ.

    If ESKF yaw has error Δψ = ψ_true − ψ_eskf, the corrected velocity is:
        v_corr = R(Δψ) @ v_eskf
    where R is the 2-D rotation matrix.
    """
    c, s = np.cos(dpsi), np.sin(dpsi)
    return c * vx - s * vy, s * vx + c * vy


def dead_reckon(pos0: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Euler-integrate velocity from pos0 at 50 Hz.

    Returns position array (T, 2) where pos[0] == pos0.
    """
    T   = len(vx)
    pos = np.empty((T, 2), dtype=np.float32)
    pos[0] = pos0
    for t in range(1, T):
        pos[t, 0] = pos[t-1, 0] + vx[t-1] * DT
        pos[t, 1] = pos[t-1, 1] + vy[t-1] * DT
    return pos


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_flight(fid: str) -> dict:
    print(f'\n{"─"*60}')
    print(f'Flight: {fid}')
    print(f'{"─"*60}')

    feat, eskf, gt_pos, gt_yaw, gt_vel = load_flight(fid)
    T         = len(feat)
    duration  = T * DT
    print(f'  Steps: {T:,}  ({duration/60:.1f} min)')

    yaw_eskf = eskf[:, 5]
    vx_eskf  = eskf[:, 6]
    vy_eskf  = eskf[:, 7]
    eskf_pos = eskf[:, 0:2]   # actual sensor-fusion position (FAST-LIO/baro/sonar)

    # ── Model predictions ────────────────────────────────────────────────────
    dpsi  = predict_yaw_correction(feat)          # (T,) rad
    dvel  = predict_velocity_correction(feat)     # (T, 2) m/s

    yaw_corr = yaw_eskf + dpsi

    # Config B: Yaw FF only — rotate world-frame velocity by Δψ
    vx_B, vy_B = rotate_velocity(vx_eskf, vy_eskf, dpsi)

    # Config C: Velocity LSTM on RAW ESKF velocity.
    # Trained on raw ESKF features → predicts (gt_vel - eskf_vel_raw).
    # Applied on raw ESKF, not yaw-corrected, to avoid double-counting yaw error.
    vx_C = vx_eskf + dvel[:, 0]
    vy_C = vy_eskf + dvel[:, 1]

    # ── Dead reckoning (GPS-denied scenario) ─────────────────────────────────
    # IMPORTANT: gt_pos is published at ~10-17 Hz (3-9 identical rows at 50 Hz
    # logging rate). Comparing ∫eskf_vel against raw gt_pos inflates error ~4×.
    # Correct reference: integrate gt_vel from the same starting point (pos0).
    # Both dead-reckoned trajectories then share the same physics.
    pos0       = gt_pos[0]
    pos_gt_dr  = dead_reckon(pos0, gt_vel[:, 0], gt_vel[:, 1])   # GT DR reference
    pos_A      = dead_reckon(pos0, vx_eskf, vy_eskf)
    pos_B      = dead_reckon(pos0, vx_B,    vy_B)
    pos_C      = dead_reckon(pos0, vx_C,    vy_C)

    # ── ESKF sensor-fusion position RMSE vs Gazebo gt_pos ────────────────────
    # gt_pos is subsampled (~10-17 Hz) but still a valid absolute reference for
    # the sensor-fusion position maintained by FAST-LIO/baro/sonar.
    eskf_pos_rmse = float(np.sqrt(np.mean(np.sum((eskf_pos - gt_pos)**2, axis=1))))

    # ── Yaw RMSE ─────────────────────────────────────────────────────────────
    yaw_err_A = np.degrees(np.abs(_wrap(yaw_eskf - gt_yaw)))
    yaw_err_B = np.degrees(np.abs(_wrap(yaw_corr  - gt_yaw)))

    yaw_rmse_A = float(np.sqrt(np.mean(yaw_err_A**2)))
    yaw_rmse_B = float(np.sqrt(np.mean(yaw_err_B**2)))

    # ── Velocity RMSE ────────────────────────────────────────────────────────
    def vel_rmse(vx_est, vy_est):
        evx   = float(np.sqrt(np.mean((gt_vel[:, 0] - vx_est)**2)))
        evy   = float(np.sqrt(np.mean((gt_vel[:, 1] - vy_est)**2)))
        ecomb = float(np.sqrt(np.mean((gt_vel[:, 0] - vx_est)**2 +
                                      (gt_vel[:, 1] - vy_est)**2)))
        return evx, evy, ecomb

    vrmse_A = vel_rmse(vx_eskf, vy_eskf)
    vrmse_B = vel_rmse(vx_B,    vy_B)
    vrmse_C = vel_rmse(vx_C,    vy_C)

    # ── Position error vs GT dead-reckoning reference ─────────────────────────
    def pos_err_at(pos_est, t_sec):
        """Euclidean position error vs pos_gt_dr at t_sec from start [m]."""
        idx = min(int(t_sec / DT), T - 1)
        return float(np.linalg.norm(pos_est[idx] - pos_gt_dr[idx]))

    def pos_rmse_full(pos_est):
        return float(np.sqrt(np.mean(np.sum((pos_est - pos_gt_dr)**2, axis=1))))

    def drift_rate(pos_est):
        """Linear fit on Euclidean error vs ∫gt_vel → drift rate [m/min]."""
        err   = np.linalg.norm(pos_est - pos_gt_dr, axis=1)
        t_m   = np.arange(T) * DT / 60.0
        slope, _ = np.polyfit(t_m, err, 1)
        return float(max(slope, 0.0))

    snapshots = {}
    for cfg_name, pos_est in [('A', pos_A), ('B', pos_B), ('C', pos_C)]:
        snaps = {f'{t}s': pos_err_at(pos_est, t) for t in SNAPSHOT_TIMES}
        snaps['end']                  = pos_err_at(pos_est, duration)
        snaps['rmse_full']            = pos_rmse_full(pos_est)
        snaps['drift_rate_m_per_min'] = drift_rate(pos_est)
        snapshots[cfg_name] = snaps

    # Theoretical upper bound: velocity RMSE × total time
    theoretical_max = vrmse_A[2] * duration

    # ── Print per-flight summary ──────────────────────────────────────────────
    print(f'\n  ESKF sensor-fusion position RMSE: {eskf_pos_rmse:.2f} m  (vs Gazebo gt_pos)')
    print(f'  Dead-reckoning bound: {theoretical_max:.1f} m  '
          f'(= {vrmse_A[2]:.3f} m/s × {duration:.0f} s)')

    print(f'\n  Yaw RMSE:')
    print(f'    A  ESKF only : {yaw_rmse_A:6.2f}°')
    print(f'    B  + Yaw FF  : {yaw_rmse_B:6.2f}°  ({100*(yaw_rmse_A-yaw_rmse_B)/yaw_rmse_A:+.1f}%)')

    print(f'\n  Velocity RMSE (vx, vy, combined):')
    print(f'    A  ESKF only      : {vrmse_A[0]:.3f}  {vrmse_A[1]:.3f}  {vrmse_A[2]:.3f} m/s')
    print(f'    B  + Yaw FF       : {vrmse_B[0]:.3f}  {vrmse_B[1]:.3f}  {vrmse_B[2]:.3f} m/s  '
          f'({100*(vrmse_A[2]-vrmse_B[2])/vrmse_A[2]:+.1f}%)')
    print(f'    C  + Vel LSTM(raw): {vrmse_C[0]:.3f}  {vrmse_C[1]:.3f}  {vrmse_C[2]:.3f} m/s  '
          f'({100*(vrmse_A[2]-vrmse_C[2])/vrmse_A[2]:+.1f}%)')

    print(f'\n  Dead-reckoning error vs ∫gt_vel (velocity-bounded reference):')
    hdr = f'    {"Config":<22}' + ''.join(f'{t}s'.rjust(8) for t in SNAPSHOT_TIMES) \
          + '  End(m)  RMSE(m)  Drift(m/min)'
    print(hdr)
    for cfg, label in [('A', 'A  ESKF only'), ('B', 'B  + Yaw FF'), ('C', 'C  + Vel LSTM(raw)')]:
        s    = snapshots[cfg]
        cols = ''.join(f'{s[f"{t}s"]:7.1f} ' for t in SNAPSHOT_TIMES)
        print(f'    {label:<22}{cols}  {s["end"]:6.1f}   {s["rmse_full"]:6.1f}   '
              f'{s["drift_rate_m_per_min"]:.2f}')

    return {
        'duration_min':    duration / 60.0,
        'eskf_pos_rmse_m': eskf_pos_rmse,
        'yaw_rmse_deg':    {'A': yaw_rmse_A, 'B': yaw_rmse_B},
        'vel_rmse_mps': {
            'A': {'vx': vrmse_A[0], 'vy': vrmse_A[1], 'combined': vrmse_A[2]},
            'B': {'vx': vrmse_B[0], 'vy': vrmse_B[1], 'combined': vrmse_B[2]},
            'C': {'vx': vrmse_C[0], 'vy': vrmse_C[1], 'combined': vrmse_C[2]},
        },
        'pos_snapshots': snapshots,
        '_arrays': {
            'T': T, 'gt_pos': gt_pos, 'gt_vel': gt_vel,
            'eskf_pos':  eskf_pos,             # sensor-fusion position
            'pos_gt_dr': pos_gt_dr,            # ∫gt_vel — correct DR reference
            'pos_A': pos_A, 'pos_B': pos_B, 'pos_C': pos_C,
            'vx_A': vx_eskf, 'vy_A': vy_eskf,
            'vx_B': vx_B,    'vy_B': vy_B,
            'vx_C': vx_C,    'vy_C': vy_C,
            'yaw_eskf': yaw_eskf, 'yaw_corr': yaw_corr, 'gt_yaw': gt_yaw,
        },
    }


# ── Run all val flights ────────────────────────────────────────────────────────

results = {}
for fid in VAL_FLIGHTS:
    try:
        results[fid] = eval_flight(fid)
    except FileNotFoundError as e:
        print(f'\nWARNING: {e} — skipping')

if not results:
    print('\nNo results — check DATA_DIR and VAL_FLIGHTS list.')
    raise SystemExit(1)


# ── Aggregate summary ─────────────────────────────────────────────────────────

configs = ['A', 'B', 'C']
labels  = {
    'A': 'A  ESKF only       ',
    'B': 'B  + Yaw FF        ',
    'C': 'C  + Vel LSTM(raw) ',
}

print(f'\n{"═"*65}')
print('AGGREGATE SUMMARY — all val flights (mean ± std)')
print(f'{"═"*65}')

print('\nYaw RMSE (deg):')
for cfg in ['A', 'B']:
    vals = [r['yaw_rmse_deg'][cfg] for r in results.values()]
    print(f'  {labels[cfg]}: {np.mean(vals):.2f}° ± {np.std(vals):.2f}°')

print('\nVelocity RMSE combined (m/s):')
for cfg in configs:
    vals = [r['vel_rmse_mps'][cfg]['combined'] for r in results.values()]
    imp  = [(r['vel_rmse_mps']['A']['combined'] - r['vel_rmse_mps'][cfg]['combined'])
            / r['vel_rmse_mps']['A']['combined'] * 100 for r in results.values()]
    print(f'  {labels[cfg]}: {np.mean(vals):.3f} ± {np.std(vals):.3f} m/s  '
          f'({np.mean(imp):+.1f}% vs A)')

print('\nPosition drift rate (m/min):')
for cfg in configs:
    vals = [r['pos_snapshots'][cfg]['drift_rate_m_per_min'] for r in results.values()]
    imp  = [(r['pos_snapshots']['A']['drift_rate_m_per_min'] -
             r['pos_snapshots'][cfg]['drift_rate_m_per_min'])
            / max(r['pos_snapshots']['A']['drift_rate_m_per_min'], 1e-3) * 100
            for r in results.values()]
    print(f'  {labels[cfg]}: {np.mean(vals):.2f} ± {np.std(vals):.2f} m/min  '
          f'({np.mean(imp):+.1f}% vs A)')


# ── Save JSON (strip _arrays before serialising) ──────────────────────────────

def _serialise(r):
    return {k: v for k, v in r.items() if k != '_arrays'}

OUT_JSON.write_text(json.dumps({fid: _serialise(r) for fid, r in results.items()}, indent=2))
print(f'\nResults saved to {OUT_JSON}')


# ── Plots ─────────────────────────────────────────────────────────────────────

colors = {'A': '#888888', 'B': '#2196F3', 'C': '#E91E63', 'GT_DR': '#4CAF50'}

n_flights = len(results)
fig, axes = plt.subplots(n_flights, 3, figsize=(18, 6 * n_flights))
if n_flights == 1:
    axes = axes[np.newaxis, :]

W = int(10.0 / DT)    # 10-second rolling window

for row, (fid, res) in enumerate(results.items()):
    arr       = res['_arrays']
    T         = arr['T']
    t_min     = np.arange(T) * DT / 60.0
    gt_p      = arr['gt_pos']
    gt_v      = arr['gt_vel']
    eskf_p    = arr['eskf_pos']
    pos_gt_dr = arr['pos_gt_dr']

    # ── Col 0: XY — GT vs ESKF sensor-fusion (actual flight footprint) ─────────
    # Dead-reckoned paths (pos_A/B/C) are NOT shown here.
    # Although velocity is world-frame, the drone's circuit/back-and-forth pattern
    # means total path length ≈ 3.9× net displacement. Integrating velocity traces
    # ~70 km of path over 29 min while the GT footprint is ~4×3 km — incompatible
    # scales that make XY overlay meaningless.
    # GPS-denied navigation quality is shown correctly in Col 2 (error vs time).
    ax = axes[row, 0]
    ax.plot(gt_p[:, 0],   gt_p[:, 1],   'k-',  lw=1.8,
            label='GT  (Gazebo pos)', alpha=0.85)
    ax.plot(eskf_p[:, 0], eskf_p[:, 1], '--',  lw=1.1, color=colors['A'],
            label=f'ESKF fusion  RMSE = {res["eskf_pos_rmse_m"]:.1f} m', alpha=0.9)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'{fid}\nESKF sensor-fusion position vs GT')
    ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    # ── Col 1: Velocity error A / B / C with 10 s rolling mean ──────────────────
    ax = axes[row, 1]
    for vx_k, vy_k, cfg in [('vx_A', 'vy_A', 'A'), ('vx_B', 'vy_B', 'B'), ('vx_C', 'vy_C', 'C')]:
        ve      = np.sqrt((arr[vx_k] - gt_v[:, 0])**2 + (arr[vy_k] - gt_v[:, 1])**2)
        ve_roll = np.convolve(ve, np.ones(W) / W, mode='valid')
        t_roll  = t_min[W-1:]
        rmse    = res['vel_rmse_mps'][cfg]['combined']
        ax.plot(t_roll, ve_roll, color=colors[cfg], lw=1.1,
                label=f'{cfg}  RMSE = {rmse:.3f} m/s')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('|Δv| combined (m/s)')
    ax.set_title('Velocity error — 10 s rolling mean\nA vs B (+Yaw FF) vs C (+Vel LSTM)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Col 2: GPS-denied dead-reckoning error vs ∫gt_vel ──────────────────────
    # pos_A/B/C all start at gt_pos[0] and integrate their respective velocities.
    # Reference = pos_gt_dr = ∫gt_vel (same integration, ideal velocity).
    # Error is therefore bounded by velocity RMSE × T (consistent with Col 1).
    ax = axes[row, 2]
    for cfg, pk in [('A', 'pos_A'), ('B', 'pos_B'), ('C', 'pos_C')]:
        err = np.linalg.norm(arr[pk] - pos_gt_dr, axis=1)
        dr  = res['pos_snapshots'][cfg]['drift_rate_m_per_min']
        ax.plot(t_min, err, color=colors[cfg], lw=0.9,
                label=f'{cfg}  {dr:.0f} m/min drift')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Dead-reckoning error (m)')
    ax.set_title('GPS-denied DR error vs ∫gt_vel\n(velocity-bounded — path length ≈ 3.9× displacement)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('MARID ESKF Cascade Correction Evaluation\n'
             'A = ESKF baseline   B = +Yaw FF   C = +Vel LSTM (raw ESKF inputs)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches='tight')
print(f'\nPlot saved to {OUT_PLOT}')
plt.close()
