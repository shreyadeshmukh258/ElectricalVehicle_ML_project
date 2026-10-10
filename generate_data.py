"""Builds EV_Predictive_Maintenance_Dataset.csv: original columns + new battery/brake/tire/coolant readings,
with realistic wear -> reading links and labels (component condition, failure, remaining life)."""
import numpy as np, pandas as pd
from features import SPEC, comp_risk, risk
rng = np.random.default_rng(7); N = 20000
B = lambda a, b: rng.beta(a, b, N); G = lambda s: rng.normal(0, s, N)
w = B(1.4, 5); mix = lambda: np.clip(.5 * w + .5 * B(1.4, 4.5), 0, 1)   # wear: vehicle-wide + per system
lb, lm, lk, lt = mix(), mix(), mix(), mix()
amb = rng.uniform(-5, 42, N); load = 300 + 1500 * B(2, 3); spd = rng.uniform(0, 150, N); rough = np.clip(B(2, 5) + .3 * lt, 0, 1)
d = pd.DataFrame(dict(
 SoC=rng.uniform(5, 100, N), SoH=100 - 45 * lb + G(3), Battery_Temperature=22 + .3 * amb + 24 * lb + .004 * load + G(4),
 Battery_Voltage=398 - 80 * lb ** 1.2 + G(6), Battery_Current=30 + 130 * B(2, 4) + .03 * load + 20 * (spd / 150),
 Charge_Cycles=1500 * np.clip(.7 * lb + .3 * B(2, 4), 0, 1) ** 1.1, Internal_Resistance=30 + 85 * lb ** 1.2 + G(5),
 Cell_Imbalance=6 + 120 * lb ** 1.4 + G(7), Motor_Temperature=50 + 70 * lm + .4 * amb + 25 * (spd / 150) + G(5),
 Motor_Vibration=.8 + 7.5 * lm ** 1.6 + np.abs(G(.5)), Motor_Torque=60 + 330 * B(2, 3) + 40 * lm, Coolant_Temperature=38 + 60 * lm + .3 * amb + G(4),
 Brake_Pad_Wear=100 * np.clip(1.25 * lk + G(.07), 0, 1), Brake_Pressure=35 + 75 * lk + 20 * B(2, 4) + G(4),
 Reg_Brake_Efficiency=93 - 68 * lk + G(4), Brake_Fluid_Level=100 - 95 * lk ** 1.2 + G(5),
 Tire_Pressure=33 + G(1 + 5 * lt) - 3 * lt, Tire_Temperature=26 + .3 * amb + 34 * lt + 8 * (spd / 150) + G(3),
 Tire_Tread_Depth=8.5 - 7 * np.clip(.85 * lt + .15 * w, 0, 1) + G(.4), Suspension_Load=60 + 120 * lt + .22 * load + G(30),
 Route_Roughness=rough, Load_Weight=load, Ambient_Temperature=amb, Driving_Speed=spd, Vehicle_Age=w * 12 + G(.8)))
for s in SPEC: d[s.key] = d[s.key].clip(s.lo, s.hi)
cr = {}
for c in ['Battery', 'Motor', 'Brakes', 'Tires']:
    cr[c] = comp_risk(np.array([risk(s, d[s.key]) for s in SPEC if s.group == c])) + G(.03)
    d[c + '_Condition'] = np.digitize(cr[c], [.25, .5])          # 0 Good, 1 Fair, 2 Poor
R = np.array(list(cr.values())); overall = .5 * R.max(0) + .5 * R.mean(0)
d['Failure'] = (rng.random(N) < 1 / (1 + np.exp(-10 * (overall - .5)))).astype(int)
d['RUL'] = (450 * np.clip(1 - overall, 0, 1) ** 1.5 + G(10)).clip(0, 450).round(1)
d.insert(0, 'Timestamp', pd.date_range('2023-01-01', periods=N, freq='15min')); d.to_csv('EV_Predictive_Maintenance_Dataset.csv', index=False)
print(d.shape, 'failure rate', d.Failure.mean().round(3))
for c in ['Battery', 'Motor', 'Brakes', 'Tires']: print(c, d[c + '_Condition'].value_counts(normalize=True).sort_index().round(2).to_dict())
