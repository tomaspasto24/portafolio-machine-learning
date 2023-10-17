import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

data = pd.read_csv('Titanic.csv', delimiter=';')

data = data[['Age', 'No of Parents or Children on Board', 'No of Siblings or Spouses on Board',
             'Passenger Fare', 'Port of Embarkation', 'Sex', 'Survived']]

data = pd.get_dummies(data, columns=['Port of Embarkation', 'Sex'], prefix='Class')


data['Age'].fillna(data['Age'].mean(), inplace=True)

X = data.drop('Survived', axis=1)
y = data['Survived']

model = DecisionTreeClassifier(criterion='gini', max_depth=9, min_samples_split=4)

scores = cross_val_score(model, X, y, cv=10)

print(scores)

model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

print("Precisión en validación cruzada:", scores.mean())
print("Precisión en todo el conjunto de datos:", accuracy)