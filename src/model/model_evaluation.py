import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

with open('models/model.pkl', 'rb') as fobj:
    clf = pickle.load(fobj)


test_data = pd.read_csv("./data/interim/test_bow.csv")

X_test = test_data.iloc[:,:-1].values
y_test = test_data.iloc[:,-1].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict = {
    "accuracy":accuracy,
    "precision":precision,
    "recall":recall,
    "auc": auc
}

with open("reports/metrics.json", "w") as fobj:
    json.dump(metrics_dict, fobj, indent=4)