"""
model_generator.py
Author: Diego Lopez
Date: 04/05/2022
This file contains the code to generate and pickle models used in analysis. 
The purpose of this is to be able to run models on cloud and not have to keep a jupyter server open. 
Note that all these set to use max cores available, please adjust if using on local machine. 
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
import pickle 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
# same geneva scale from playlist_generator.py
# except we added a "_" for in love, as this was added at export time
geneva_scale = {"Wonder" : ["Happy", "Dazzling", "Alluring"], "Transcendence": ["Inspiring", "Spiritual"], "Tenderness":
        ["In_love", 'Sensual'], "Nostalgia": ["Nostalgic", "Sentimental", "Dreamy"], "Peacefulness": ["Peaceful", "Calm", "Relaxing"],
    "Power" : ["Energetic", "Fiery", "Heroic"], "Joyful" : ["Joyful", "Dancing"], "Tension" : ["Agitated", "Nervous"], "Sadness" : ["Sad", "Sorrow"]
}
# path to directory of csvs
complete_df = pd.read_csv("../data/processed/complete_dataset.csv", index_col=0)
print("[INFO] Data loaded...")
complete_df.info()
complete_df = complete_df.dropna()
scaler = StandardScaler()
# fit to the numeric columns
scaler.fit(complete_df.iloc[:, np.r_[0:11, 16:18]])
# scale the dataset and reassign columns
# keep as np array for better performance
scaled_complete_df = scaler.transform(complete_df.iloc[:, np.r_[0:11, 16:18]])
print("[INFO] Data normalized...")
# create train test split
X = scaled_complete_df
y = complete_df["emotion"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle= True, test_size=0.25)
cv = KFold(n_splits=5, shuffle=True)
# logistic regression
clf_lr = LogisticRegressionCV(Cs=10, fit_intercept=False, cv=cv, penalty="elasticnet", solver="saga", max_iter=1000, l1_ratios=[0,0.25, 0.5, 0.75, 1], verbose=1, n_jobs=-1)
print("[INFO] Starting Logistic Regression...")
clf_lr.fit(X_train, y_train)
with open("../models/clf_lr.pkl", "wb") as f:
    pickle.dump(clf_lr, f)
print("[INFO] Completed Logistic Regression...")
# Random Forest
rf = RandomForestClassifier(n_jobs=-1)
params = {
    # using sqrt max features as auto
    "max_samples" : [None, 0.32, 0.68],
    "n_estimators" : [250, 500],
    "max_depth" : [None, 50, 75],
    "criterion" : ["gini"],
    "min_samples_split" : [20, 25]
}
print("[INFO] Starting Random Forest...")
clf_rf = GridSearchCV(rf, param_grid=params, scoring="accuracy", n_jobs=-1, cv=cv, verbose=1)
clf_rf.fit(X_train, y_train)
with open("../models/clf_rf.pkl", "wb") as f:
    pickle.dump(clf_rf, f)
print("Completed Random Forest...")
# Ada boost
abc = AdaBoostClassifier(random_state=0)
params = {
    "n_estimators" : [250, 500, 1000],
    "learning_rate" : [0.1, 0.5, 1]
}
clf_abc = GridSearchCV(abc, param_grid=params, scoring="accuracy", n_jobs=-1, cv=cv, verbose=1)
print("[INFO] Starting Ada Boost...")
clf_abc.fit(X_train, y_train)
with open("../models/clf_abc.pkl", "wb") as f:
    pickle.dump(clf_abc, f)
print("Completed Ada boost...")
# KN Classifier
knc = KNeighborsClassifier(n_jobs = -1)
params = {
    "n_neighbors" : [5, 10, 20, 50, 100],
    "algorithm" : ["ball_tree", "kd_tree"],
    "leaf_size" : [40],
    "p" : [1,2],
}
clf_knc = GridSearchCV(knc, param_grid=params, scoring="accuracy", cv=cv, verbose=1)
print("[INFO] Starting KN Classifier...")
clf_knc.fit(X_train, y_train)
with open("../models/clf_knc.pkl", "wb") as f:
    pickle.dump(clf_knc, f)
print("Completed KN Classifier...")
# Decision Tree
dtc = DecisionTreeClassifier()
params = {
    "criterion" : ["gini", "entropy"],
    "max_depth" : [None, 50],
    "max_features" : ["sqrt", None]
}
clf_dtc = GridSearchCV(dtc, param_grid=params, scoring="accuracy", n_jobs=-1, cv=cv, verbose=1)
print("[INFO] Starting Decision Tree...")
clf_dtc.fit(X_train, y_train)
with open("../models/clf_dtc.pkl", "wb") as f:
    pickle.dump(clf_dtc, f)
print("Completed Decision Tree...")
# Gradient Boosting, this one takes the longest!
gbc = GradientBoostingClassifier()
params = {
    "loss" : ["deviance"],
    "learning_rate" : [0.05, 0.1],
    "n_estimators" : [200, 400, 450],
    "max_features" : ["sqrt"],
    "max_depth" : [10, 15]        
}
clf_gbc = GridSearchCV(gbc, param_grid=params, scoring="accuracy", n_jobs=-1, cv=cv, verbose=1)
print("[INFO] Starting Gradient Boosting...")
clf_gbc.fit(X_train, y_train)
with open("../models/clf_gbc.pkl", "wb") as f:
    pickle.dump(clf_gbc, f)
print("Completed Gradient Boosting...")
# MLP Classifier
mlp = MLPClassifier(verbose=1)
params = {
    "hidden_layer_sizes" : [(10,10,10), (10, 10), (10)],
    "activation" : ["relu"], 
    "solver" : ["adam"],
    "alpha" : [0.0001, 0.001],
    "learning_rate" : ["adaptive", "constant"],
    "batch_size" : [64, 128, 256], # similar to changing learning rate and with less param updates
    "learning_rate_init" : [0.001, 0.01],
}
clf_mlp = GridSearchCV(mlp, param_grid=params, scoring="accuracy", n_jobs=-1, cv=cv, verbose=2)
print("[INFO] Starting MLP Classifier...")
clf_mlp.fit(X_train, y_train)
with open("../models/clf_mlp.pkl", "wb") as f:
    pickle.dump(clf_mlp, f)
print("Completed MLP Classifier...")
print("Completed all models successfully, script exiting!")