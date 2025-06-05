import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

import yaml

config = yaml.safe_load(open('params.yaml', 'r'))

n_estimators = config['model_building']['n_estimators']
lr = config['model_building']['learning_rate']

train_data = pd.read_csv("./data/interim/train_bow.csv")

X_train = train_data.iloc[:,:-1].values
y_train = train_data.iloc[:,-1].values

clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr)
clf.fit(X_train, y_train)


with open("models/model.pkl", "wb") as fobj:
    pickle.dump(clf, fobj)
