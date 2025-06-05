import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import os

import yaml

config = yaml.safe_load(open('params.yaml', 'r'))

max_features = config['feature_engineering']['max_features']


train_data = pd.read_csv("./data/processed/train_processed.csv")
test_data = pd.read_csv("./data/processed/test_processed.csv")

train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

vectorizer = CountVectorizer(max_features=max_features)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

data_path = os.path.join("data", "interim")

os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test_bow.csv"),index=False)