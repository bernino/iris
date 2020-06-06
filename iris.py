import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import plotly.express as px


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

X = iris.data
y = iris.target
df['target'] = y
print(df.head())

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

corr_matrix = df.corr().abs()
print(corr_matrix['target'].sort_values(ascending=False).head(11))

# first let's run all the data and setup models
# to find a model that works the best
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
resultsmean = []
resultsstddev = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    resultsmean.append(cv_results.mean())
    resultsstddev.append(cv_results.std())

resultsDf = pd.DataFrame(
    {'name': names,
     'mean': resultsmean,
     'std dev': resultsstddev
    }
)
resultsDf = resultsDf.sort_values(by=['mean'], ascending=False)
print(resultsDf)

# Make predictions using validation dataset
model1 = SVC(gamma='auto')
model1.fit(X_train, Y_train)
predictions = model1.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))