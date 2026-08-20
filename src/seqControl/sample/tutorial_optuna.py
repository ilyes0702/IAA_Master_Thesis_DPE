import seaborn as sns
import pandas as pd
healthexp = sns.load_dataset("healthexp")



healthexp = pd.get_dummies(healthexp)

X = healthexp.drop(["Life_Expectancy"], axis = 1)
y = healthexp["Life_Expectancy"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=34)

rfr.fit(X_train, y_train    )

y_pred = rfr.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(mean_absolute_error(y_test, y_pred))


import optuna

from sklearn.model_selection import cross_val_score

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 10, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 32)
    min_samples_leaf = trial.suggest_int("n_estimmin_samples_leafators", 1, 32)

    model = RandomForestRegressor(n_estimators=n_estimators, 
                                  max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs = -1).mean()

    return score

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42))

study.optimize(objective, n_trials=200)

print(study.best_params)

best_params = study.best_params

import matplotlib.pyplot as plt

