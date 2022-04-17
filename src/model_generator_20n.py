"""
model_generator_10n.py
Author: Diego Lopez
Date: 04/05/2022
This file contains the code to generate and pickle models used in analysis. 
This file specifically deals with running models sampled with 20% additional noise injected
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
# turn X back into a dataframe to use native sampling method
full_scaled_df = pd.DataFrame(X_train)
full_scaled_df["emotion"] = y_train.to_numpy()

noisy_sample_df_20 = pd.DataFrame()
for key in geneva_scale.keys(): 
    print(key)
    length = len(full_scaled_df[full_scaled_df["emotion"] == key])
    # sample from correct labeled distribution
    noisy_sample_df_20 = pd.concat([noisy_sample_df_20, full_scaled_df[full_scaled_df["emotion"] == key].sample(frac=0.8, replace=False)])
    # sample from noisy distribution with wrong labels
    noisy_sample_df_20 = pd.concat([noisy_sample_df_20, full_scaled_df[full_scaled_df['emotion'] != key].sample(n=int(0.2*length), replace=False).assign(emotion=key)])

full_scaled_df.dropna(inplace=True)
noisy_sample_df_20.dropna(inplace=True)
print("length of dataset with every label having 20% noise" ,len(noisy_sample_df_20))
print("length of weakly supervised dataset", len(full_scaled_df))
print("resampling off")

X_train_20n = noisy_sample_df_20.iloc[:,:-1]
y_train_20n = noisy_sample_df_20.iloc[:,-1]

# Random Forest
clf_rf_20n = RandomForestClassifier(criterion="gini", max_depth=50, min_samples_split=20, n_estimators=250, n_jobs=-1)
print("[INFO] Starting Random Forest...")
clf_rf_20n.fit(X_train_20n, y_train_20n)
with open("clf_rf_20n.pkl", "wb") as f:
    pickle.dump(clf_rf_20n, f)

# Logistic Regression
clf_lr_20n = LogisticRegressionCV(Cs=10, fit_intercept=False, cv=cv, penalty="elasticnet", solver="saga", max_iter=1000, l1_ratios=[0,0.25, 0.5, 0.75, 1], verbose=1, n_jobs=-1)
print("[INFO] Starting Logistic Regression...")
clf_lr_20n.fit(X_train_20n, y_train_20n)

with open("clf_lr_20n.pkl", "wb") as f:
    pickle.dump(clf_lr_20n, f)

# Ada Boost
clf_abc_20n = AdaBoostClassifier(random_state=0, learning_rate=0.5, n_estimators=500)
print("[INFO] Starting Ada Boost...")
clf_abc_20n.fit(X_train_20n, y_train_20n)
with open("clf_abc_20n.pkl", "wb") as f:
    pickle.dump(clf_abc_20n, f)

# Decision Tree
clf_dtc_20n = DecisionTreeClassifier(criterion="gini", max_depth=None, max_features=None)
print("[INFO] Starting Decision Tree...")
clf_dtc_20n.fit(X_train_20n, y_train_20n)
with open("clf_dtc_20n.pkl", "wb") as f:
    pickle.dump(clf_dtc_20n, f)

# KN Classifier
clf_knc_20n = KNeighborsClassifier(algorithm="ball_tree", leaf_size=40, n_neighbors=20, p=1)
print("[INFO] Starting KN Classifier...")
clf_knc_20n.fit(X_train_20n, y_train_20n)
with open("clf_knc_20n.pkl", "wb") as f:
    pickle.dump(clf_knc_20n, f)

# Gradient Boost
clf_gbc_20n = GradientBoostingClassifier(learning_rate=0.1, loss="deviance", max_depth=10, max_features="sqrt", n_estimators=400)
print("[INFO] Starting Gradient Boost...")
clf_gbc_20n.fit(X_train_20n, y_train_20n)
with open("clf_gbc_20n.pkl", "wb") as f:
    pickle.dump(clf_gbc_20n, f)

# MLP Classifier
clf_mlp_20n = MLPClassifier(activation="relu", alpha=0.0001, batch_size=128, hidden_layer_sizes=(10,10,10), 
                            learning_rate="adaptive", learning_rate_init=0.001, solver="adam")
print("[INFO] Starting MLP Classifier...")
clf_mlp_20n.fit(X_train_20n, y_train_20n)
with open("clf_mlp_20n.pkl", "wb") as f:
    pickle.dump(clf_mlp_20n, f)