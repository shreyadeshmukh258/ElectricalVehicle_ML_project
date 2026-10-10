"""Tests 5 algorithms on all 6 outputs, then uses the ONE algorithm with the best average score for everything."""
import pandas as pd, numpy as np, json, joblib, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import *
from features import KEYS
df = pd.read_csv('EV_Predictive_Maintenance_Dataset.csv'); X = df[KEYS]; sc = lambda m: make_pipeline(StandardScaler(), m)
ALGOS = {  # name: (classifier, regressor)
 'Linear model': (lambda: sc(LogisticRegression(max_iter=2000)), lambda: sc(Ridge())),
 'Decision Tree': (lambda: DecisionTreeClassifier(max_depth=8, random_state=0), lambda: DecisionTreeRegressor(max_depth=8, random_state=0)),
 'KNN': (lambda: sc(KNeighborsClassifier(15)), lambda: sc(KNeighborsRegressor(15))),
 'Random Forest': (lambda: RandomForestClassifier(150, min_samples_leaf=3, n_jobs=-1, random_state=0), lambda: RandomForestRegressor(100, min_samples_leaf=3, n_jobs=-1, random_state=0)),
 'Gradient Boosting': (lambda: HistGradientBoostingClassifier(random_state=0), lambda: HistGradientBoostingRegressor(random_state=0))}
targets = {c: (c + '_Condition', 0) for c in ['Battery', 'Motor', 'Brakes', 'Tires']}; targets.update({'Failure': ('Failure', 0), 'RUL': ('RUL', 1)})
scores = {a: {} for a in ALGOS}
for t, (col, reg) in targets.items():
    cv = KFold(3, shuffle=True, random_state=0) if reg else StratifiedKFold(3, shuffle=True, random_state=0)
    for a, mk in ALGOS.items():
        scores[a][t] = round(cross_validate(mk[reg](), X, df[col], cv=cv, scoring='r2' if reg else 'f1_macro')['test_score'].mean(), 4)
avg = {a: round(float(np.mean(list(s.values()))), 4) for a, s in scores.items()}
best = max(avg, key=avg.get)
for a in ALGOS: print(f'{a:18s} avg={avg[a]:.4f}', scores[a])
print('\nUSING:', best)
models = {t: ALGOS[best][reg]().fit(X, df[col]) for t, (col, reg) in targets.items()}
joblib.dump(models, 'models.joblib'); json.dump(dict(chosen=best, average=avg, scores=scores), open('report.json', 'w'), indent=1)
