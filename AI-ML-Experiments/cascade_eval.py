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
  New logs carry per-sample dt from eskf_gt_logger. For those logs, all
  dead-reckoning integrations use dt directly and compare against Gazebo gt_pos.
  Older logs have no timing sidecar, so their DR metrics are marked as fixed-DT
  fallback and should be treated as approximate only.

Metrics per config per flight:
  - Yaw RMSE (deg)
  - Velocity RMSE: vx, vy, combined (m/s)
  - Dead-reckoning position RMSE at 30 s, 60 s, 120 s, 300 s, end-of-flight (m)
  - ESKF sensor-fusion position RMSE vs Gazebo gt_pos (separate from DR errors)

Output: console table + cascade_eval_results.json + cascade_eval_plots.png in DATA_DIR.
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

DATA_DIR      = Path('~/marid_ws/data_sync').expanduser()      # flight chunks (val flights)
MODEL_DIR     = Path('~/marid_ws/data_sync').expanduser()      # velocity LSTM + position LSTM
MODEL_DIR_FF  = Path('~/marid_ws/data_sync').expanduser()      # attitude FF (retrained on data_sync)
OUT_JSON  = DATA_DIR / 'cascade_eval_results.json'
OUT_PLOT  = DATA_DIR / 'cascade_eval_plots.png'

DT        = 1.0 / 50.0   # fallback only for legacy logs without dt [s]
CHUNK_LEN = 300           # LSTM inference chunk (must match training CHUNK_LEN)
MAX_SOURCE_AGE_SEC = 0.030

# Val flights held out from velocity LSTM training (cold-start, no data leakage)
VAL_FLIGHTS = [
    'flight_20260601_063117',
    'flight_20260601_071638',
    'flight_20260601_074510',
    'flight_20260601_082638',
    'flight_20260601_090950',
    'flight_20260603_211100',
    'flight_20260604_064124',
    'flight_20260604_074357',
    'flight_20260604_084209',
    'flight_20260604_094326',
    'flight_20260604_104717',
    'flight_20260604_115444',
    'flight_20260605_063323',
]

# Snapshot times for position error table [s]
SNAPSHOT_TIMES = [30, 60, 120, 300]

# ── Helpers ───────────────────────────────────────────────────────────────────

_G        = 9.81
_MIN_V    = 3.0
_PDOT_MAX = 2.0


def _wrap(a):
    return ((a + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def _legacy_timebase(n: int):
    dt = np.full(n, DT, dtype=np.float32)
    t = np.cumsum(dt, dtype=np.float64).astype(np.float32)
    return t, dt, 'legacy_fixed_dt'


def _chunk_timebase(d: dict, n: int):
    if 'dt' in d:
        dt = d['dt'].astype(np.float32).ravel()[:n].copy()
        dt[~np.isfinite(dt)] = 0.0
        dt = np.maximum(dt, 0.0)
        if 't' in d:
            t = d['t'].astype(np.float64).ravel()[:n].copy()
            if len(t) == n and np.all(np.isfinite(t)):
                # Some early sync logs stored wall-clock t but source-stamp dt.
                # Prefer dt-derived source time whenever the sidecars disagree.
                if n < 2 or abs(float(t[-1] - t[0]) - float(np.sum(dt))) <= max(0.5, 0.05 * float(np.sum(dt))):
                    return t.astype(np.float32), dt, 'logged_dt'
                return np.cumsum(dt, dtype=np.float64).astype(np.float32), dt, 'logged_dt_rebuilt_t'
        return np.cumsum(dt, dtype=np.float64).astype(np.float32), dt, 'logged_dt'
    return _legacy_timebase(n)


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
    gt_vel : (T, 2)  float32   [vx_gt, vy_gt] m/s
    t      : (T,)    float32   seconds from flight start
    dt     : (T,)    float32   seconds since previous sample, dt[0] = 0
    meta   : dict              optional source-stamp diagnostics
    """
    chunks = sorted(DATA_DIR.glob(f'marid_eskf_gt_{fid}_chunk*.npz'))
    if not chunks:
        raise FileNotFoundError(f'No chunks found for {fid} in {DATA_DIR}')

    feat_parts, eskf_parts, gt_pos_parts, gt_yaw_parts, gt_vel_parts = [], [], [], [], []
    t_parts, dt_parts, time_sources = [], [], set()
    eskf_stamp_parts, gt_stamp_parts = [], []
    for p in chunks:
        d    = np.load(p, allow_pickle=True)
        eskf = d['eskf_inputs'].astype(np.float32)
        gt   = d['pose_targets'].astype(np.float32)   # (N, 7): [x,y,roll,pitch,yaw,vx,vy]
        n    = len(eskf)
        t_chunk, dt_chunk, time_source = _chunk_timebase(d, n)

        feat_parts.append(_augment_eskf_inputs(eskf, d))
        eskf_parts.append(eskf)
        gt_pos_parts.append(gt[:, 0:2])
        gt_yaw_parts.append(gt[:, 4])
        gt_vel_parts.append(gt[:, 5:7])
        t_parts.append(t_chunk)
        dt_parts.append(dt_chunk)
        time_sources.add(time_source)
        if 'eskf_stamp_sec' in d and 'gt_stamp_sec' in d:
            eskf_stamp_parts.append(d['eskf_stamp_sec'].astype(np.float64).ravel()[:n])
            gt_stamp_parts.append(d['gt_stamp_sec'].astype(np.float64).ravel()[:n])

    dt = np.concatenate(dt_parts)
    if len(dt):
        dt[0] = 0.0
    if len(time_sources) == 1 and 'logged_dt' in time_sources:
        t = np.concatenate(t_parts)
        t = (t - t[0]).astype(np.float32)
    else:
        # Legacy chunks each start from their own local fallback timebase; rebuild
        # a single continuous vector after concatenation.
        t = np.cumsum(dt, dtype=np.float64).astype(np.float32)

    meta = {}
    if len(eskf_stamp_parts) == len(chunks) and len(gt_stamp_parts) == len(chunks):
        eskf_stamp = np.concatenate(eskf_stamp_parts)
        gt_stamp = np.concatenate(gt_stamp_parts)
        age = eskf_stamp - gt_stamp
        fresh_gt = np.ones(len(gt_stamp), dtype=bool)
        if len(gt_stamp) > 1:
            fresh_gt[1:] = np.abs(np.diff(gt_stamp)) > 1.0e-9
        meta['eskf_stamp_sec'] = eskf_stamp
        meta['gt_stamp_sec'] = gt_stamp
        meta['source_stamp_age_sec'] = age
        meta['matched_velocity_mask'] = (
            np.isfinite(age) &
            (np.abs(age) <= MAX_SOURCE_AGE_SEC) &
            fresh_gt
        )

    return (np.concatenate(feat_parts),
            np.concatenate(eskf_parts),
            np.concatenate(gt_pos_parts),
            np.concatenate(gt_yaw_parts),
            np.concatenate(gt_vel_parts),
            t,
            dt,
            '+'.join(sorted(time_sources)),
            meta)


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


class PositionLSTM(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=128, num_layers=1, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(self.drop(out)), hidden


# ── Load models ───────────────────────────────────────────────────────────────

print('Loading models...')

# Attitude FF (loaded from data_low_err — not yet retrained on data_sync)
_ff_norm  = np.load(MODEL_DIR_FF / 'eskf_attitude_ff_norm.npz')
ff_X_mean = _ff_norm['X_mean'].astype(np.float32)
ff_X_std  = _ff_norm['X_std'].astype(np.float32)
ff_y_mean = float(_ff_norm['y_mean'])
ff_y_std  = float(_ff_norm['y_std'])

att_model = make_attitude_ff()
att_model.load_state_dict(torch.load(MODEL_DIR_FF / 'eskf_attitude_ff.pt', map_location='cpu', weights_only=True))
att_model.eval()
print(f'  Attitude FF loaded  (y_mean={ff_y_mean:.4f} rad, y_std={ff_y_std:.4f} rad)')

# Velocity LSTM
_vel_norm  = np.load(MODEL_DIR / 'eskf_velocity_lstm_norm.npz')
vel_X_mean = _vel_norm['X_mean'].astype(np.float32)
vel_X_std  = _vel_norm['X_std'].astype(np.float32)
vel_y_mean = _vel_norm['y_mean'].astype(np.float32)   # (2,)
vel_y_std  = _vel_norm['y_std'].astype(np.float32)    # (2,)

vel_model = VelocityLSTM()
vel_model.load_state_dict(torch.load(MODEL_DIR / 'eskf_velocity_lstm.pt', map_location='cpu', weights_only=True))
vel_model.eval()
print(f'  Velocity LSTM loaded (y_mean={vel_y_mean}, y_std={vel_y_std})')

# Position LSTM (optional — skip gracefully if not yet trained)
_pos_model_path = MODEL_DIR / 'eskf_position_lstm.pt'
_pos_norm_path  = MODEL_DIR / 'eskf_position_lstm_norm.npz'
pos_model = None
pos_X_mean = pos_X_std = pos_y_mean = pos_y_std = None
if _pos_model_path.exists() and _pos_norm_path.exists():
    _pos_norm  = np.load(_pos_norm_path)
    pos_X_mean = _pos_norm['X_mean'].astype(np.float32)
    pos_X_std  = _pos_norm['X_std'].astype(np.float32)
    pos_y_mean = _pos_norm['y_mean'].astype(np.float32)
    pos_y_std  = _pos_norm['y_std'].astype(np.float32)
    pos_model  = PositionLSTM()
    pos_model.load_state_dict(torch.load(_pos_model_path, map_location='cpu', weights_only=True))
    pos_model.eval()
    print(f'  Position LSTM loaded (y_std={pos_y_std})')
else:
    print(f'  Position LSTM not found — Config D will be skipped')


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


def predict_position_correction(feat: np.ndarray) -> np.ndarray:
    """[Δx, Δy] position correction for every timestep [m].

    Predicts (gt_pos - eskf_fused_pos) — corrects residual ESKF sensor-fusion
    position error (FAST-LIO / baro / sonar drift). Not a GPS-denied dead-reckoning
    correction; the ESKF position input must come from a running sensor-fusion stack.
    """
    if pos_model is None:
        return None
    T      = len(feat)
    delta  = np.zeros((T, 2), dtype=np.float32)
    X_norm = ((feat - pos_X_mean) / pos_X_std).astype(np.float32)
    hidden = None
    with torch.no_grad():
        for i in range(0, T, CHUNK_LEN):
            chunk = torch.tensor(X_norm[i:i + CHUNK_LEN]).unsqueeze(0)
            pred, hidden = pos_model(chunk, hidden)
            hidden = (hidden[0].detach(),
                      torch.clamp(hidden[1].detach(), -100.0, 100.0))
            raw = pred.squeeze(0).numpy()
            delta[i:i + len(raw)] = raw * pos_y_std + pos_y_mean
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


def dead_reckon(pos0: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                dt: np.ndarray) -> np.ndarray:
    """Euler-integrate velocity from pos0 using per-sample dt.

    Returns position array (T, 2) where pos[0] == pos0.
    """
    T   = len(vx)
    pos = np.empty((T, 2), dtype=np.float32)
    pos[0] = pos0
    for t in range(1, T):
        pos[t, 0] = pos[t-1, 0] + vx[t-1] * dt[t]
        pos[t, 1] = pos[t-1, 1] + vy[t-1] * dt[t]
    return pos


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_flight(fid: str) -> dict:
    print(f'\n{"─"*60}')
    print(f'Flight: {fid}')
    print(f'{"─"*60}')

    feat, eskf, gt_pos, gt_yaw, gt_vel, t_sec, dt_sec, time_source, meta = load_flight(fid)
    T         = len(feat)
    duration  = float(t_sec[-1]) if T else 0.0
    print(f'  Steps: {T:,}  ({duration/60:.1f} min)')
    print(f'  Timebase: {time_source}  (median dt={np.median(dt_sec[dt_sec > 0]):.5f}s)')
    source_age_stats = None
    if 'source_stamp_age_sec' in meta:
        age = meta['source_stamp_age_sec']
        finite_age = age[np.isfinite(age)]
        if len(finite_age):
            source_age_stats = {
                'mean_sec': float(np.mean(finite_age)),
                'median_sec': float(np.median(finite_age)),
                'p95_abs_sec': float(np.percentile(np.abs(finite_age), 95)),
                'max_abs_sec': float(np.max(np.abs(finite_age))),
            }
            print('  Source age ESKF-GT: '
                  f"mean={source_age_stats['mean_sec']*1000:+.1f} ms, "
                  f"median={source_age_stats['median_sec']*1000:+.1f} ms, "
                  f"p95|age|={source_age_stats['p95_abs_sec']*1000:.1f} ms, "
                  f"max|age|={source_age_stats['max_abs_sec']*1000:.1f} ms")
    vel_metric_mask = meta.get('matched_velocity_mask', np.ones(T, dtype=bool))
    if len(vel_metric_mask) != T or not np.any(vel_metric_mask):
        vel_metric_mask = np.ones(T, dtype=bool)
    print(f'  Velocity RMSE rows: {int(np.sum(vel_metric_mask)):,}/{T:,} '
          f'({100*np.mean(vel_metric_mask):.1f}%)')

    yaw_eskf = eskf[:, 5]
    vx_eskf  = eskf[:, 6]
    vy_eskf  = eskf[:, 7]
    eskf_pos = eskf[:, 0:2]   # actual sensor-fusion position (FAST-LIO/baro/sonar)

    # ── Model predictions ────────────────────────────────────────────────────
    dpsi  = predict_yaw_correction(feat)          # (T,) rad
    dvel  = predict_velocity_correction(feat)     # (T, 2) m/s
    dpos  = predict_position_correction(feat)     # (T, 2) m  or None

    yaw_corr = yaw_eskf + dpsi

    # Config B: Yaw FF only — rotate world-frame velocity by Δψ
    vx_B, vy_B = rotate_velocity(vx_eskf, vy_eskf, dpsi)

    # Config C: Velocity LSTM on RAW ESKF velocity.
    # Trained on raw ESKF features → predicts (gt_vel - eskf_vel_raw).
    # Applied on raw ESKF, not yaw-corrected, to avoid double-counting yaw error.
    vx_C = vx_eskf + dvel[:, 0]
    vy_C = vy_eskf + dvel[:, 1]

    # ── Dead reckoning (GPS-denied scenario) ─────────────────────────────────
    # Integrate all velocity traces with the same per-sample timing. Compare
    # against Gazebo gt_pos directly; gt_pos may be held between Gazebo updates,
    # but it is the absolute reference and avoids trusting odom twist scaling.
    pos0       = gt_pos[0]
    pos_ref    = gt_pos
    pos_A      = dead_reckon(pos0, vx_eskf, vy_eskf, dt_sec)
    pos_B      = dead_reckon(pos0, vx_B,    vy_B,    dt_sec)
    pos_C      = dead_reckon(pos0, vx_C,    vy_C,    dt_sec)

    # ── ESKF sensor-fusion position RMSE vs Gazebo gt_pos ────────────────────
    # gt_pos is subsampled (~10-17 Hz) but still a valid absolute reference for
    # the sensor-fusion position maintained by FAST-LIO/baro/sonar.
    eskf_pos_rmse = float(np.sqrt(np.mean(np.sum((eskf_pos - gt_pos)**2, axis=1))))

    # Config D: sensor-fusion position + Position LSTM correction.
    # dpos predicts (gt_pos - eskf_fused_pos); apply to the fused position, not DR.
    eskf_pos_D_rmse = None
    if dpos is not None:
        pos_D_fused    = eskf_pos + dpos
        eskf_pos_D_rmse = float(np.sqrt(np.mean(np.sum((pos_D_fused - gt_pos)**2, axis=1))))

    # ── Yaw RMSE ─────────────────────────────────────────────────────────────
    yaw_err_A = np.degrees(np.abs(_wrap(yaw_eskf[vel_metric_mask] - gt_yaw[vel_metric_mask])))
    yaw_err_B = np.degrees(np.abs(_wrap(yaw_corr[vel_metric_mask]  - gt_yaw[vel_metric_mask])))

    yaw_rmse_A = float(np.sqrt(np.mean(yaw_err_A**2)))
    yaw_rmse_B = float(np.sqrt(np.mean(yaw_err_B**2)))

    # ── Velocity RMSE ────────────────────────────────────────────────────────
    def vel_rmse(vx_est, vy_est):
        m = vel_metric_mask
        evx   = float(np.sqrt(np.mean((gt_vel[m, 0] - vx_est[m])**2)))
        evy   = float(np.sqrt(np.mean((gt_vel[m, 1] - vy_est[m])**2)))
        ecomb = float(np.sqrt(np.mean((gt_vel[m, 0] - vx_est[m])**2 +
                                      (gt_vel[m, 1] - vy_est[m])**2)))
        return evx, evy, ecomb

    vrmse_A = vel_rmse(vx_eskf, vy_eskf)
    vrmse_B = vel_rmse(vx_B,    vy_B)
    vrmse_C = vel_rmse(vx_C,    vy_C)

    # ── Position error vs GT dead-reckoning reference ─────────────────────────
    def pos_err_at(pos_est, t_sec):
        """Euclidean position error vs Gazebo gt_pos at t_sec from start [m]."""
        idx = min(int(np.searchsorted(t_sec_array, t_sec, side='left')), T - 1)
        return float(np.linalg.norm(pos_est[idx] - pos_ref[idx]))

    def pos_rmse_full(pos_est):
        return float(np.sqrt(np.mean(np.sum((pos_est - pos_ref)**2, axis=1))))

    t_sec_array = t_sec.astype(np.float64)

    snapshots = {}
    for cfg_name, pos_est in [('A', pos_A), ('B', pos_B), ('C', pos_C)]:
        snaps = {f'{t}s': pos_err_at(pos_est, t) for t in SNAPSHOT_TIMES}
        snaps['end']       = pos_err_at(pos_est, duration)
        snaps['rmse_full'] = pos_rmse_full(pos_est)
        snapshots[cfg_name] = snaps

    # Velocity RMSE × duration gives a rough scale for possible accumulated error.
    theoretical_max = vrmse_A[2] * duration

    # ── Print per-flight summary ──────────────────────────────────────────────
    print(f'\n  ESKF sensor-fusion position RMSE:')
    print(f'    A  ESKF only     : {eskf_pos_rmse:.2f} m  (vs Gazebo gt_pos)')
    if eskf_pos_D_rmse is not None:
        imp_d = 100.0 * (eskf_pos_rmse - eskf_pos_D_rmse) / eskf_pos_rmse if eskf_pos_rmse > 0 else 0.0
        print(f'    D  + Pos LSTM    : {eskf_pos_D_rmse:.2f} m  ({imp_d:+.1f}%)')
    print(f'  Velocity-error accumulation scale: {theoretical_max:.1f} m  '
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

    print(f'\n  Dead-reckoning error vs Gazebo gt_pos:')
    hdr = f'    {"Config":<22}' + ''.join(f'{t}s'.rjust(8) for t in SNAPSHOT_TIMES) \
          + '  End(m)  RMSE(m)'
    print(hdr)
    for cfg, label in [('A', 'A  ESKF only'), ('B', 'B  + Yaw FF'), ('C', 'C  + Vel LSTM(raw)')]:
        s    = snapshots[cfg]
        cols = ''.join(f'{s[f"{t}s"]:7.1f} ' for t in SNAPSHOT_TIMES)
        print(f'    {label:<22}{cols}  {s["end"]:6.1f}   {s["rmse_full"]:6.1f}')

    return {
        'duration_min':    duration / 60.0,
        'time_source':     time_source,
        'median_dt_s':     float(np.median(dt_sec[dt_sec > 0])) if np.any(dt_sec > 0) else 0.0,
        'source_age_stats': source_age_stats,
        'velocity_rmse_rows': int(np.sum(vel_metric_mask)),
        'velocity_rmse_row_fraction': float(np.mean(vel_metric_mask)),
        'eskf_pos_rmse_m': eskf_pos_rmse,
        'eskf_pos_D_rmse_m': eskf_pos_D_rmse,
        'yaw_rmse_deg':    {'A': yaw_rmse_A, 'B': yaw_rmse_B},
        'vel_rmse_mps': {
            'A': {'vx': vrmse_A[0], 'vy': vrmse_A[1], 'combined': vrmse_A[2]},
            'B': {'vx': vrmse_B[0], 'vy': vrmse_B[1], 'combined': vrmse_B[2]},
            'C': {'vx': vrmse_C[0], 'vy': vrmse_C[1], 'combined': vrmse_C[2]},
        },
        'pos_snapshots': snapshots,
        '_arrays': {
            'T': T, 't_sec': t_sec, 'dt_sec': dt_sec,
            'gt_pos': gt_pos, 'gt_vel': gt_vel,
            'eskf_pos':   eskf_pos,            # sensor-fusion position
            'eskf_pos_D': eskf_pos + dpos if dpos is not None else None,
            'pos_ref': pos_ref,                # Gazebo absolute position reference
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

print('\nESKF sensor-fusion position RMSE (m):')
pos_A_vals = [r['eskf_pos_rmse_m'] for r in results.values()]
print(f'  {"A  ESKF only       "}: {np.mean(pos_A_vals):.2f} ± {np.std(pos_A_vals):.2f} m')
pos_D_vals = [r['eskf_pos_D_rmse_m'] for r in results.values() if r['eskf_pos_D_rmse_m'] is not None]
if pos_D_vals:
    pos_A_matched = [r['eskf_pos_rmse_m'] for r in results.values() if r['eskf_pos_D_rmse_m'] is not None]
    imp_d = [(a - d) / a * 100 for a, d in zip(pos_A_matched, pos_D_vals)]
    print(f'  {"D  + Pos LSTM      "}: {np.mean(pos_D_vals):.2f} ± {np.std(pos_D_vals):.2f} m  '
          f'({np.mean(imp_d):+.1f}% vs A)')

# ── Save JSON (strip _arrays before serialising) ──────────────────────────────

def _serialise(r):
    return {k: v for k, v in r.items() if k != '_arrays'}

OUT_JSON.write_text(json.dumps({fid: _serialise(r) for fid, r in results.items()}, indent=2))
print(f'\nResults saved to {OUT_JSON}')


# ── Plots ─────────────────────────────────────────────────────────────────────

colors = {'A': '#888888', 'B': '#2196F3', 'C': '#E91E63', 'D': '#9C27B0', 'GT_DR': '#4CAF50'}

n_flights = len(results)
fig, axes = plt.subplots(n_flights, 3, figsize=(18, 6 * n_flights))
if n_flights == 1:
    axes = axes[np.newaxis, :]

for row, (fid, res) in enumerate(results.items()):
    arr       = res['_arrays']
    T         = arr['T']
    t_min     = arr['t_sec'] / 60.0
    median_dt = max(float(res.get('median_dt_s', DT)), 1.0e-6)
    W         = max(1, int(10.0 / median_dt))    # 10-second rolling window
    gt_p      = arr['gt_pos']
    gt_v      = arr['gt_vel']
    eskf_p    = arr['eskf_pos']
    pos_ref   = arr['pos_ref']

    # ── Col 0: XY — GT vs ESKF sensor-fusion (actual flight footprint) ─────────
    # Dead-reckoned paths (pos_A/B/C) are NOT shown here.
    # Dead-reckoned paths are omitted here so the absolute sensor-fusion footprint
    # stays readable. GPS-denied navigation quality is shown in Col 2.
    ax = axes[row, 0]
    ax.plot(gt_p[:, 0],   gt_p[:, 1],   'k-',  lw=1.8,
            label='GT  (Gazebo pos)', alpha=0.85)
    ax.plot(eskf_p[:, 0], eskf_p[:, 1], '--',  lw=1.1, color=colors['A'],
            label=f'A  ESKF fusion  RMSE = {res["eskf_pos_rmse_m"]:.1f} m', alpha=0.9)
    if arr.get('eskf_pos_D') is not None and res['eskf_pos_D_rmse_m'] is not None:
        ep_D = arr['eskf_pos_D']
        ax.plot(ep_D[:, 0], ep_D[:, 1], '--', lw=1.1, color='#9C27B0',
                label=f'D  + Pos LSTM   RMSE = {res["eskf_pos_D_rmse_m"]:.1f} m', alpha=0.9)
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

    # ── Col 2: GPS-denied dead-reckoning error vs Gazebo gt_pos ────────────────
    # pos_A/B/C all start at gt_pos[0] and integrate their respective velocities.
    # Reference = held Gazebo gt_pos, which may update slower than the logger.
    ax = axes[row, 2]
    for cfg, pk in [('A', 'pos_A'), ('B', 'pos_B'), ('C', 'pos_C')]:
        err = np.linalg.norm(arr[pk] - pos_ref, axis=1)
        rmse = res['pos_snapshots'][cfg]['rmse_full']
        ax.plot(t_min, err, color=colors[cfg], lw=0.9,
                label=f'{cfg}  RMSE={rmse:.1f} m')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Dead-reckoning error (m)')
    ax.set_title('GPS-denied DR error vs Gazebo gt_pos\n(timing-aware when logged dt is available)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('MARID ESKF Cascade Correction Evaluation\n'
             'A = ESKF baseline   B = +Yaw FF   C = +Vel LSTM (raw ESKF inputs)',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150, bbox_inches='tight')
print(f'\nPlot saved to {OUT_PLOT}')
plt.close()
