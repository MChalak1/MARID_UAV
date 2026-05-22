"""
MARID ESKF Model Comparison
Loads per-flight RMSE results saved by all training scripts and prints
a unified comparison table: one section per flight, one row per model.

Run after training models:
  python3 eskf_position_lstm_train.py
  python3 eskf_velocity_lstm_train.py
  python3 eskf_attitude_ff_train.py
  python3 compare_results.py
"""
import json
from pathlib import Path

DATA_DIR = Path("~/marid_ws/data").expanduser()

MODELS = [
    ('Position LSTM', 'results_eskf_position_lstm.json', ['Δx (m)', 'Δy (m)']),
    ('Velocity LSTM', 'results_eskf_velocity_lstm.json', ['Δvx (m/s)', 'Δvy (m/s)']),
    ('Attitude FF',   'results_eskf_attitude_ff.json',   ['roll (rad)', 'pitch (rad)', 'vx (m/s)', 'vy (m/s)']),
]

ALL_OUTPUTS = ['Δx (m)', 'Δy (m)', 'Δvx (m/s)', 'Δvy (m/s)', 'roll (rad)', 'pitch (rad)', 'vx (m/s)', 'vy (m/s)']

# ── Load results ──────────────────────────────────────────────────────────────

loaded = {}
for model_name, fname, _ in MODELS:
    path = DATA_DIR / fname
    if path.exists():
        loaded[model_name] = json.loads(path.read_text())
    else:
        print(f'  [missing] {fname} — run {fname.replace("results_", "").replace(".json", "_train.py")} first')
        loaded[model_name] = {}

# ── Collect all flight IDs ─────────────────────────────────────────────────────

all_flights = sorted({fid for results in loaded.values() for fid in results})
if not all_flights:
    print('No results found. Run the training scripts first.')
    exit()

# ── Print per-flight comparison tables ────────────────────────────────────────

COL = 9   # column width for RMSE values

for fid in all_flights:
    print(f'\n{"═"*80}')
    print(f'  {fid}')
    print(f'{"═"*80}')

    # Header
    hdr = f'  {"Model":<22}  {"Val MSE":>8}  ' + '  '.join(f'{lbl[:8]:>{COL}}' for lbl in ALL_OUTPUTS)
    print(hdr)
    print(f'  {"-"*22}  {"-"*8}  ' + '  '.join('-'*COL for _ in ALL_OUTPUTS))

    for model_name, _, model_outputs in MODELS:
        flight_results = loaded[model_name].get(fid)
        if flight_results is None:
            row = f'  {"  " + model_name:<22}  {"—":>8}  ' + '  '.join(f'{"—":>{COL}}' for _ in ALL_OUTPUTS)
        else:
            val_mse = flight_results['val_mse']
            rmse    = flight_results['rmse']
            cells = []
            for lbl in ALL_OUTPUTS:
                if lbl in rmse:
                    cells.append(f'{rmse[lbl]:>{COL}.3f}')
                else:
                    cells.append(f'{"—":>{COL}}')
            row = f'  {"  " + model_name:<22}  {val_mse:>8.4f}  ' + '  '.join(cells)
        print(row)

# ── Cross-flight summary: mean RMSE per model per output ──────────────────────

print(f'\n{"═"*80}')
print('  CROSS-FLIGHT MEAN RMSE (across all flights where model was trained)')
print(f'{"═"*80}')
hdr = f'  {"Model":<22}  ' + '  '.join(f'{lbl[:8]:>{COL}}' for lbl in ALL_OUTPUTS)
print(hdr)
print(f'  {"-"*22}  ' + '  '.join('-'*COL for _ in ALL_OUTPUTS))

for model_name, _, model_outputs in MODELS:
    results = loaded[model_name]
    cells = []
    for lbl in ALL_OUTPUTS:
        vals = [r['rmse'][lbl] for r in results.values() if lbl in r.get('rmse', {})]
        if vals:
            cells.append(f'{sum(vals)/len(vals):>{COL}.3f}')
        else:
            cells.append(f'{"—":>{COL}}')
    print(f'  {"  " + model_name:<22}  ' + '  '.join(cells))

print()
