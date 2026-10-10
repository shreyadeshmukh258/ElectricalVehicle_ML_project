"""Single source of truth: every reading, its range, safe limits and plain-language meaning."""
import numpy as np
from collections import namedtuple
S = namedtuple('S', 'key label unit group lo hi default mode warn bad why')
# mode: hi = high is bad, lo = low is bad, band = distance from default is bad, none = context only
SPEC = [
 S('SoC','State of charge','%','Battery',0,100,70,'lo',20,8,'Very low charge limits range and stresses the cells'),
 S('SoH','Battery health (SoH)','%','Battery',50,100,92,'lo',80,70,'Share of original capacity left; below 80% range drops noticeably'),
 S('Battery_Temperature','Battery temperature','°C','Battery',0,60,30,'hi',40,50,'Heat ages the cells faster and can trigger power limiting'),
 S('Battery_Voltage','Pack voltage','V','Battery',300,420,380,'lo',350,325,'Sagging voltage points to weak or unbalanced cells'),
 S('Battery_Current','Discharge current','A','Battery',0,250,60,'hi',160,210,'Heavy current draw heats the pack'),
 S('Charge_Cycles','Charge cycles','','Battery',0,1500,250,'hi',800,1200,'Every full charge cycle slowly wears the cells'),
 S('Internal_Resistance','Internal resistance','mΩ','Battery',20,120,40,'hi',70,95,'Rising resistance wastes energy as heat and signals aging cells'),
 S('Cell_Imbalance','Cell voltage imbalance','mV','Battery',0,150,15,'hi',50,90,'Uneven cells reduce the usable capacity of the pack'),
 S('Motor_Temperature','Motor temperature','°C','Motor',30,140,65,'hi',95,115,'An overheating motor loses power and wears its insulation'),
 S('Motor_Vibration','Motor vibration','mm/s','Motor',0,10,2,'hi',5,7.5,'Strong vibration suggests bearing wear or misalignment'),
 S('Motor_Torque','Motor torque','Nm','Motor',0,400,150,'hi',300,360,'Sustained high torque overloads the motor'),
 S('Coolant_Temperature','Coolant temperature','°C','Motor',20,110,50,'hi',75,90,'Hot coolant means the cooling loop is not keeping up'),
 S('Brake_Pad_Wear','Brake pad wear','%','Brakes',0,100,25,'hi',65,85,'Worn pads stop the car less effectively'),
 S('Brake_Pressure','Brake pressure','bar','Brakes',0,120,50,'hi',85,105,'Needing very high pressure means the brakes are fading'),
 S('Reg_Brake_Efficiency','Regenerative braking efficiency','%','Brakes',30,100,80,'lo',65,50,'Low efficiency wastes energy and loads the friction brakes'),
 S('Brake_Fluid_Level','Brake fluid level','%','Brakes',0,100,90,'lo',50,30,'Low fluid can cause a soft pedal or a leak'),
 S('Tire_Pressure','Tire pressure','psi','Tires',15,45,33,'band',4,8,'Wrong pressure hurts grip, range and tire life'),
 S('Tire_Temperature','Tire temperature','°C','Tires',10,80,33,'hi',50,65,'Hot tires wear quickly and risk a blowout'),
 S('Tire_Tread_Depth','Tire tread depth','mm','Tires',0,10,6,'lo',3,1.6,'Thin tread loses grip on wet roads (legal minimum 1.6 mm)'),
 S('Suspension_Load','Suspension load','kg','Tires',50,600,200,'hi',400,500,'Heavy load strains springs and dampers'),
 S('Route_Roughness','Road roughness','0-1','Tires',0,1,0.3,'hi',0.7,0.9,'Rough roads wear tires and suspension faster'),
 S('Load_Weight','Vehicle load','kg','Tires',200,2000,900,'hi',1500,1800,'Extra weight increases wear on tires and brakes'),
 S('Ambient_Temperature','Outside temperature','°C','Driving',-10,45,22,'none',0,0,''),
 S('Driving_Speed','Average speed','km/h','Driving',0,160,60,'none',0,0,''),
 S('Vehicle_Age','Vehicle age','years','Driving',0,15,3,'none',0,0,''),
]
KEYS = [s.key for s in SPEC]
COMPONENTS = {
 'Battery': dict(good='The battery is in good shape and holds charge well.', fair='The battery is aging. Expect slightly less range and slower charging.', poor='The battery is worn or stressed. Range loss and sudden power cuts are likely, so get it checked soon.'),
 'Motor':   dict(good='The motor runs cool and smooth.', fair='The motor is warmer or rougher than normal. Check cooling and mounts at the next service.', poor='The motor is overheating or vibrating heavily. Reduce load and service it soon to avoid a breakdown.'),
 'Brakes':  dict(good='Brakes have plenty of life and normal braking force.', fair='Brakes are wearing. Plan a pad or fluid service.', poor='Braking performance is reduced. Service the brakes before driving much further.'),
 'Tires':   dict(good='Tires and suspension are in good shape.', fair='Tires or suspension show wear or stress. Check pressure and tread.', poor='Tires or suspension are worn or overloaded. Repair or replace them soon.')}
ACTION = ['No service needed', 'Plan a check at the next service', 'Service as soon as possible']

def risk(s, x):
    """0 = fully healthy, 0.5 = at the warning limit, 1 = at the critical limit."""
    x = np.asarray(x, float)
    if s.mode == 'none': return np.zeros_like(x)
    v = np.abs(x - s.default) if s.mode == 'band' else x
    return np.clip(.5 + .5 * (v - s.warn) / (s.bad - s.warn), 0, 1)

def comp_risk(r):  # r = array of feature risks for one component; one bad reading matters
    return .5 * r.max(0) + .5 * r.mean(0)

def good_bad(s):  # slider values used by the presets
    if s.mode == 'none': return s.default, s.default
    if s.mode == 'band': return s.default, s.default - s.bad
    return float(np.clip(s.warn - (s.bad - s.warn), s.lo, s.hi)), s.bad
