"""Quick check: estimates all 25 readings from 6 things an owner knows. Rules mirror how generate_data.py links wear to readings."""
import numpy as np
from features import SPEC
def estimate(age=3, odo=40000, range_drop=8, brake_months=6, tread=None, st=.5, amb=22, soc=70, tire_psi=None, warn=()):
    c = lambda x: float(np.clip(x, 0, 1));
    w = lambda n: .25 if n in warn else 0   # a warning light pushes that system's wear up
    lb = c(range_drop / 45 * .8 + age / 12 * .2 + w('Battery')); lm = c(.5 * age / 12 + .5 * odo / 250000 + w('Motor'))
    lk = c(.5 * odo / 150000 + .5 * brake_months / 48 + .15 * st + w('Brakes')); lt = c(min(1, odo / 90000 + w('Tires'))) if tread is None else c((8.5 - tread) / 7 + w('Tires'))
    spd, load = 40 + 40 * st, 900
    v = dict(SoC=soc, SoH=100 - range_drop, Battery_Temperature=22 + .3 * amb + 24 * lb, Battery_Voltage=398 - 80 * lb ** 1.2,
      Battery_Current=45 + 60 * st + .03 * load, Charge_Cycles=odo / 250, Internal_Resistance=30 + 85 * lb ** 1.2, Cell_Imbalance=6 + 120 * lb ** 1.4,
      Motor_Temperature=50 + 70 * lm + .4 * amb + 25 * spd / 150, Motor_Vibration=.8 + 7.5 * lm ** 1.6, Motor_Torque=100 + 120 * st + 40 * lm,
      Coolant_Temperature=38 + 60 * lm + .3 * amb, Brake_Pad_Wear=100 * c(1.25 * lk), Brake_Pressure=35 + 75 * lk + 10 * st,
      Reg_Brake_Efficiency=93 - 68 * lk, Brake_Fluid_Level=100 - 95 * lk ** 1.2, Tire_Pressure=tire_psi if tire_psi else 33 - 3 * lt,
      Tire_Temperature=26 + .3 * amb + 34 * lt + 8 * spd / 150, Tire_Tread_Depth=8.5 - 7 * lt if tread is None else tread,
      Suspension_Load=60 + 120 * lt + .22 * load, Route_Roughness=.3, Load_Weight=load, Ambient_Temperature=amb, Driving_Speed=spd, Vehicle_Age=age)
    return {s.key: float(np.clip(v[s.key], s.lo, s.hi)) for s in SPEC}
