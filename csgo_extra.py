import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('./datasets/csgo.csv')
data = df.drop(columns=['team_a_rounds', 'team_b_rounds', 'date'])
encoder = LabelEncoder()
data['map'] = encoder.fit_transform(data['map'])
data['result'] = encoder.fit_transform(data['result'])
scaler = StandardScaler()
scaler.fit_transform(data)
corr = data.corr()
