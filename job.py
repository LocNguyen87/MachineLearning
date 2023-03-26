import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_excel('./datasets/final_project.ods', engine='odf', dtype=str)
target = 'career_level'
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
result = vectorizer.fit_transform(data["title"])
print(vectorizer.vocabulary_)
print(result[0])
