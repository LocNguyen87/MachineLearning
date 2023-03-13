import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
df = pd.read_csv('./datasets/csgo.csv')

# drop the team rounds columns
data = df.drop(columns=['team_a_rounds', 'team_b_rounds'])

# define the target
target = "result"

# explore the dataset
# print(df.info())

# encode the categorical variables
encoder = LabelEncoder()
data['map'] = encoder.fit_transform(data['map'])
data[target] = encoder.fit_transform(data[target])

# split the dataset
x_train, x_test, y_train, y_test = \
    train_test_split(data.drop(columns=[target, 'date']), data[target], test_size=0.2, random_state=42)

# scale the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# create the pipelines to compare the regression models
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', scaler),
                                        ('LR', LogisticRegression())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', scaler),
                                        ('RF', RandomForestClassifier())])))

def modeling(models):
    for name, model in models:
        kfold = KFold(n_splits=10)
        results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
        print(f'{name} = {results.mean()}')


# model basic comparison by accuracy score
modeling(pipelines)

# create & fine tune LogisticRegression model
params_lr = {
    "tol": [1e-4, 1e-3, 1e-2, 1e-1],
    "C": [1.0, 0.1, 0.01, 0.001],
    "solver": ['liblinear', 'saga'],
    "max_iter": [100, 1000, 2000, 3000, 4000]
}

lr = GridSearchCV(LogisticRegression(max_iter=4000), params_lr, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
lr.fit(x_train, y_train)
y_predict_lr = lr.predict(x_test)

# create & fine tune RandomForestClassifier model
params_rf = {
    "n_estimators": [50, 100, 150, 200],
    "criterion": ['gini', 'entropy', 'log_loss'],
    "max_depth": [None, 1, 5, 10],
    "max_features": ['auto', 'sqrt', 'log2']
}

rf = GridSearchCV(RandomForestClassifier(), params_rf, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
rf.fit(x_train, y_train)
y_predict_rf = rf.predict(x_test)

# print the results
print(lr.best_params_)
print(rf.best_params_)
print('LogisticRegression result:')
print(classification_report(y_test, y_predict_lr))
print('RandomForestClassifier result:')
print(classification_report(y_test, y_predict_rf))
# print(confusion_matrix(y_test, y_predict))

# plot the confusion matrix
cm_lr = np.array(confusion_matrix(y_test, y_predict_lr, labels=[0, 1, 2]))
cm_rf = np.array(confusion_matrix(y_test, y_predict_rf, labels=[0, 1, 2]))
corr_lr = pd.DataFrame(cm_lr, index=['lost', 'tie', 'win'],
                       columns=['predicted lost', 'predicted tie', 'predicted win'])
corr_rf = pd.DataFrame(cm_rf, index=['lost', 'tie', 'win'],
                       columns=['predicted lost', 'predicted tie', 'predicted win'])
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].set_title('LogisticRegression Confusion Matrix')
ax[1].set_title('RandomForestClassifier Confusion Matrix')
sns.heatmap(corr_lr, annot=True, cmap='Blues', fmt='g', ax=ax[0])
sns.heatmap(corr_rf, annot=True, cmap='Blues', fmt='g', ax=ax[1])
plt.show()
# save the plot
fig.savefig('./csgo_confusion_matrix_comparison.png')
