import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt

data = pd.read_csv('./datasets/diabetes.csv')

target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    "n_estimators": [50, 100, 150, 200],
    "criterion": ['gini', 'entropy', 'log_loss'],
    "max_depth": [None, 1, 5, 10],
    "max_features": ['auto', 'sqrt', 'log2']
}

cls = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))
