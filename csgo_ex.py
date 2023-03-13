import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
df = pd.read_csv('./datasets/csgo.csv')

# drop the team rounds columns
data = df.drop(columns=['team_a_rounds', 'team_b_rounds'])

# define the target
target = "result"

# explore the dataset
print(df.info())

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

# create & fine tune prediction model
# params = {
#     "n_estimators": [50, 100, 150, 200],
#     "criterion": ['gini', 'entropy', 'log_loss'],
#     "max_depth": [None, 1, 5, 10],
#     "max_features": ['auto', 'sqrt', 'log2']
# }
#
# cls = GridSearchCV(RandomForestClassifier(), params, cv=5, n_jobs=-1, verbose=1)

# create model based on best params from GridSearchCV
cls = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='sqrt', n_estimators=100)
cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
# print(cls.best_params_)

# print the results
# print(classification_report(y_test, y_predict))
# print(confusion_matrix(y_test, y_predict))

# plot the confusion matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[0, 1, 2]))
corr = pd.DataFrame(cm, index=['lost', 'tie', 'win'],
                    columns=['predicted lost', 'predicted tie', 'predicted win'])
sns.heatmap(corr, annot=True, cmap='Blues', fmt='g')

# save the plot
plt.savefig('csgo_predict_heatmap.png')
