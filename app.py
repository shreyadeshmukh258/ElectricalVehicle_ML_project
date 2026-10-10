import io, json, joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request, Response
from features import SPEC, KEYS, COMPONENTS, ACTION, risk, good_bad
from estimate import estimate
app = Flask(__name__)
models = joblib.load('models.joblib'); report = json.load(open('report.json'))
NAMES = ['Good', 'Fair', 'Poor']; BY = {s.key: s for s in SPEC}; DEF = {s.key: s.default for s in SPEC}

def issues_for(group, v):
    out = []
    for s in (s for s in SPEC if s.group == group and s.mode != 'none'):
        r = float(risk(s, v[s.key]))
        if r >= .5:
            lim = f'{s.default - s.warn:g} to {s.default + s.warn:g}' if s.mode == 'band' else f'{"above" if s.mode == "hi" else "below"} {s.warn:g}'
            out.append(dict(key=s.key, label=s.label, value=f'{v[s.key]:g} {s.unit}'.strip(), status='bad' if r >= .95 else 'warn', limit=lim, why=s.why, r=r))
    return sorted(out, key=lambda i: -i['r'])

def frame(v): return pd.DataFrame([[v[k] for k in KEYS]], columns=KEYS)
def cond(c, v): return int(models[c].predict(frame(v))[0])

def tip(c, v, k, iss):
    """Smallest change that would improve the rating, found by re-running the model."""
    if k == 0 or not iss: return None
    for i in iss:
        gv = good_bad(BY[i['key']])[0]; nk = cond(c, {**v, i['key']: gv})
        if nk < k: return f"If {i['label'].lower()} were back to normal ({gv:g} {BY[i['key']].unit}), the {c.lower()} would move from {NAMES[k]} to {NAMES[nk]}.".replace('  ', ' ')
    fixed = {**v, **{i['key']: good_bad(BY[i['key']])[0] for i in iss}}; nk = cond(c, fixed)
    return f"Fixing all flagged readings would move the {c.lower()} from {NAMES[k]} to {NAMES[nk]}." if nk < k else None

def analyze(v, note=None, trend=None):
    X = frame(v); comps = []
    for c, info in COMPONENTS.items():
        p = models[c].predict_proba(X)[0]; k = int(p.argmax()); iss = issues_for(c, v)
        comps.append(dict(name=c, condition=NAMES[k], confidence=round(float(p[k]) * 100), score=round(float(p @ [100, 55, 10])),
            message=info[['good', 'fair', 'poor'][k]], action=ACTION[k], issues=iss, tip=tip(c, v, k, iss)))
    pf = float(models['Failure'].predict_proba(X)[0, 1]); rul = float(np.clip(models['RUL'].predict(X)[0], 0, 450))
    worst = max(comps, key=lambda c: (NAMES.index(c['condition']), -c['score']))
    summary = 'All four systems look healthy. Keep to the regular service schedule.' if worst['condition'] == 'Good' else f"Focus on the {worst['name'].lower()} first: {worst['action'].lower()}."
    return dict(components=comps, failure=dict(prob=pf, level='High' if pf >= .6 else 'Medium' if pf >= .3 else 'Low'),
                rul_days=round(rul), summary=summary, readings=v, note=note, trend=trend)

def err(msg, code=400): return jsonify(error=msg), code

@app.get('/')
def home(): return render_template('index.html')

@app.get('/api/meta')
def meta(): return jsonify(report=report, features=[dict(s._asdict(), good=good_bad(s)[0], badval=good_bad(s)[1]) for s in SPEC])

@app.post('/api/predict')
def predict():
    d = request.get_json(force=True, silent=True) or {}
    try: v = {s.key: float(d.get(s.key, s.default)) for s in SPEC}
    except (TypeError, ValueError): return err('All readings must be numbers')
    return jsonify(analyze(v))

LAB = dict(age='vehicle age', odo='odometer', soc='battery level', range_shown='range shown', range_new='range when new', amb='outside temperature', tire_psi='tire pressure', cons='energy use', brake_months='brake service time')
LIM = dict(age=(0, 30), odo=(0, 600000), soc=(0, 100), range_shown=(1, 1000), range_new=(50, 1000), amb=(-30, 55), tire_psi=(10, 60), cons=(5, 40), brake_months=(0, 120))

@app.post('/api/check')
def check():
    d = request.get_json(force=True, silent=True) or {}; a = {}
    for k, (lo, hi) in LIM.items():
        if d.get(k) in (None, ''): continue
        try: a[k] = float(d[k])
        except (TypeError, ValueError): return err(f'The {LAB[k]} must be a number.')
        if not lo <= a[k] <= hi: return err(f'The {LAB[k]} looks unrealistic. Please re-check it.')
    kw = {}
    if 'age' in a: kw['age'] = a['age']
    if 'odo' in a: kw['odo'] = a['odo']
    if 'amb' in a: kw['amb'] = a['amb']
    if 'brake_months' in a: kw['brake_months'] = a['brake_months']
    if 'tire_psi' in a: kw['tire_psi'] = a['tire_psi']
    if 'cons' in a: kw['st'] = float(np.clip((a['cons'] - 12) / 12, 0, 1))
    kw['warn'] = [w for w in d.get('warn', []) if w in COMPONENTS]
    notes = []
    if a.get('soc', 0) >= 10 and 'range_shown' in a and 'range_new' in a:   # range left vs. new, corrected for cold weather
        full = a['range_shown'] / (a['soc'] / 100) * (1 + max(0, 10 - a.get('amb', 20)) * .01)
        kw['range_drop'] = float(np.clip(100 * (1 - full / a['range_new']), 0, 50)); kw['soc'] = a['soc']
        notes.append(f"Your car shows about {full:.0f} km at full charge, {kw['range_drop']:.0f}% below new.")
    miss = [LAB[k] for k in LAB if k not in a]
    if miss: notes.append('Typical values were used for what you skipped: ' + ', '.join(miss) + '.')
    notes.append('Sensor values you cannot see on the screen were estimated, so treat this as guidance, not a diagnosis.')
    return jsonify(analyze(estimate(**kw), ' '.join(notes)))

@app.post('/api/upload')
def upload():
    f = request.files.get('file')
    if not f or not f.filename.lower().endswith('.csv'): return err('Please choose a .csv file.')
    try: df = pd.read_csv(f)
    except Exception: return err('That file could not be read as a CSV.')
    low = {k.lower(): k for k in KEYS}; df.columns = [low.get(c.strip().lower(), c) for c in df.columns]
    have = [k for k in KEYS if k in df.columns]
    if not have: return err('No known columns found. Download the sample file to see the expected column names.')
    df[have] = df[have].apply(pd.to_numeric, errors='coerce')
    if df[have].isna().all().any() or len(df) < 1: return err('Some columns contain no valid numbers.')
    for k in KEYS: df[k] = df[k].fillna(df[k].median()) if k in have else DEF[k]
    df = df.tail(200); T = models['Failure'].predict_proba(df[KEYS])[:, 1]
    step = max(1, len(df) // 40); v = {k: float(df[k].tail(10).mean()) for k in KEYS}
    miss = [BY[k].label for k in KEYS if k not in have]
    note = f'Used the average of your last {min(10, len(df))} rows. ' + (f'{len(miss)} readings were missing and use typical values: {", ".join(miss[:6])}{"..." if len(miss) > 6 else ""}.' if miss else 'All readings were present.')
    return jsonify(analyze(v, note, [round(float(x), 3) for x in T[::step]]))

@app.get('/sample.csv')
def sample():
    rng = np.random.default_rng(1); n = 60; f = np.linspace(.05, .6, n); rows = {}
    for s in SPEC:
        g, b = good_bad(s); rows[s.key] = np.clip(g + (b - g) * f + rng.normal(0, (s.hi - s.lo) * .006, n), s.lo, s.hi).round(2)
    out = pd.DataFrame(rows); out.insert(0, 'Timestamp', pd.date_range('2025-01-01', periods=n, freq='D'))
    return Response(out.to_csv(index=False), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=sample_vehicle.csv'})

if __name__ == '__main__': app.run(debug=True)
