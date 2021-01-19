# -*- coding: utf-8 -*-
"""Stacking per il dataset NIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14Klbrz2y_BeO8HnElRD0IdKDTBi8I-gg

# **Stacking per il dataset NIST**

## Introduzione

### **Stacking**

Per un'introduzione all'*ensemble learning* e al *bagging* si può far riferimento al seguente [notebook](https://colab.research.google.com/drive/1_NvI3EKvWB-X60bA4krSC6d-IRP0TQBl#scrollTo=_SoVElADZO6h).

Lo *stacking* differisce dal *bagging* e dal *boosting* principalmente su due punti:
* nello stacking i weak learner possono essere *eterogenei*, ovvero, vengono combinati diversi algoritmi di apprendimento. Nel bagging e nel boosting si considerano soltanto dei weak learner *omogenei*;
* lo stacking impara a combinare i weak learner utilizzando un **meta-modello**. Nel bagging e nel boosting i weak learner vengono combinati seguendo algoritmi deterministici.

Nei punti precedenti sono già state definite le caratteristiche dello stacking. Volendo sintetizzare, l'idea dello stacking è quella di apprendere dei weak learner eterogenei e combinarli addestrando un meta-modello per produrre delle predizioni basate sulle predizione multiple restituite da tali weak learner.

Come nel bagging, anche nello stacking è possibile addestrare contemporaneamente i vari weak learner.

### **Dataset**

Il dataset utilizzato in questo notebook viene fornito da sklearn. In alternativa, il dataset è reperibile al seguente link: [https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits).

## Caricamento del dataset e delle librerie necessarie
"""

import numpy as np
import pandas as pd

# Dataset
from sklearn.datasets import load_digits

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Splitting
from sklearn.model_selection import train_test_split

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Ensemble
from sklearn.ensemble import StackingClassifier

# Cross Validation
from sklearn.model_selection import GridSearchCV, KFold

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score 
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
from tabulate import tabulate

dataset = load_digits()

"""## Esplorazione del dataset

Si ottenga una descrizione del dataset.
"""

print(dataset.DESCR)

"""## Split del dataset"""

X = dataset.data
y = dataset.target

"""Si suddivida il dataset in training set e test set."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## Feature scaling

Si effettui uno scaling dei dati con `StandardScaler`.
"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""## Model selection

Si definiscano i classificatori che verranno utilizzati nella model selection.
"""

cls_names = ["Decision Tree", 
             "K-Nearest Neighbors",
             "Logistic Regression",
             "Neural Network"] 

classifiers = [DecisionTreeClassifier,
               KNeighborsClassifier,
               LogisticRegression,
               MLPClassifier]

"""Si crei un dizionario di dizionari che contiene la griglia dei parametri per i classificatori."""

p_grid = {
    "dt"  : [{
              'max_depth': [None, 3, 5, 7, 9], 
              'splitter': ['best', 'random'], 
              'criterion': ['gini', 'entropy'], 
              'ccp_alpha': [0, 0.001, 0.01, 0.1], 
              'random_state': [42]
              }],
    "knn" : [{
              'n_neighbors': [3, 5, 7, 9], 
              'weights': ['uniform', 'distance'], 
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 
              'leaf_size': [20, 30, 40], 
              'n_jobs': [-1]
              }],
    "lgr" : [{
              'C':[0.001, 0.01, 0.1, 1, 2, 4], 
              'max_iter':[100, 200, 500, 1000], 
              'warm_start': [True, False]
              }],
    "nn"  : [{
              'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (250,)], 
              'learning_rate_init': [0.001, 0.01, 0.1], 
              'learning_rate': ['constant', 'adaptive'], 
              'solver': ['adam', 'sgd'], 
              'max_iter': [200, 1000],
              'batch_size': ['auto', 32, 64, 128],
              'random_state': [42]
            }]
}
p_grid

"""Si definisca K-Fold Cross Validation con K = 5."""

skf = KFold(n_splits=5, shuffle=False, random_state=42)

best_models, best_scores = [], []

for i, (name, clf, grid) in enumerate(zip(cls_names, classifiers, p_grid)):
  print("\nALGORITHM: ", name)

  fold, models, scores = 1, [], []

  for train, test in skf.split(X, y):
    print("\nFOLD:", fold)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train])
    X_test = scaler.transform(X[test])

    gridCV = GridSearchCV(clf(), param_grid=p_grid[grid], cv=5, scoring='accuracy', n_jobs=-1)
    gridCV.fit(X_train, y[train].ravel())

    print("VALIDATION score:", gridCV.best_score_)
    print("BEST parameters:", gridCV.best_params_)

    y_pred = gridCV.predict(X_test)
    y_test = y[test]

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print("TEST score:", str(acc))
    print()

    models.append({"name": name,
                   "validation_score":  gridCV.best_score_,
                   "params":            gridCV.best_params_,
                   "test_score":        str(acc),
                   "report":            classification_report(y_test, y_pred),
                   "confusion_matrix":  confusion_matrix(y_test, y_pred)})
    
    scores.append(gridCV.best_score_)

    fold += 1

  best_model = models[np.argmax(scores)]
  best_score = scores[np.argmax(scores)]

  print("\nBEST MODEL\n")
  print("Validation score: ", best_model["validation_score"])
  print("Params: ",           best_model["params"])
  print("Test score: ",       best_model["test_score"])
  print("Report\n",           best_model["report"])
  print("Confusion matrix\n", best_model["confusion_matrix"])

  best_models.append(best_model)
  best_scores.append(best_score)

"""Si illustri il modello che ha performato meglio nella model selction."""

best_model = best_models[np.argmax(best_scores)]

print("Name: ",             best_model["name"])
print("Validation score: ", best_model["validation_score"])
print("Params: ",           best_model["params"])
print("Test score: ",       best_model["test_score"])
print("Report\n",           best_model["report"])
print("Confusion matrix\n", best_model["confusion_matrix"])

"""Si illustrino i modelli che hanno performato meglio nella model selction."""

for best_model in best_models:
  print("Name: ",             best_model["name"])
  print("Validation score: ", best_model["validation_score"])
  print("Params: ",           best_model["params"])
  print("Test score: ",       best_model["test_score"])
  print("Report\n",           best_model["report"])
  print("Confusion matrix\n", best_model["confusion_matrix"])

"""## Test dei classificatori"""

X = dataset.data
y = dataset.target

"""Si suddivida il dataset in training set e test set."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Si effettui uno scaling dei dati con `StandardScaler`."""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""In questa sezione, verranno testati i modelli ottenuti nella fase di model selection."""

dt = DecisionTreeClassifier(max_depth=None, criterion='entropy', ccp_alpha=0, splitter='random', random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

knn = KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=20, weights='uniform', n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)

nn = MLPClassifier(hidden_layer_sizes=(200,), 
                    batch_size=64, 
                    learning_rate='constant', 
                    learning_rate_init=0.001, 
                    max_iter=200, 
                    solver='adam', 
                    random_state=42)
nn.fit(X_train, y_train)
y_pred = nn.predict(X_test)
accuracy_score(y_test, y_pred)

lgr = LogisticRegression(C=0.1, max_iter=100, warm_start=True, random_state=42, n_jobs=-1)
lgr.fit(X_train, y_train)
y_pred = lgr.predict(X_test)
accuracy_score(y_test, y_pred)

"""## Stacking

Si utilizzino come **weak learner** *DecisionTree*, *K-Nearest Neighbors* e una *rete neurale*.
"""

weak_learners = []
weak_learners.append(('dt', DecisionTreeClassifier(max_depth=None, criterion='entropy', ccp_alpha=0, splitter='random', random_state=42)))
weak_learners.append(('knn', KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=20, weights='uniform', n_jobs=-1)))
weak_learners.append(('nn', MLPClassifier(hidden_layer_sizes=(200,), learning_rate_init=0.001, learning_rate='constant', solver='adam', max_iter=200, batch_size=64, random_state=42)))

"""Come **meta-modello** si utilizzi una *LogisticRegression*."""

meta_model = LogisticRegression(C=0.1, max_iter=100, warm_start=True, random_state=42, n_jobs=-1)

"""Si definisca un *ensemble* di classificatori con `StackingClassifier`."""

model = StackingClassifier(estimators=weak_learners, final_estimator=meta_model, n_jobs=-1)

"""Si addestri il modello."""

model.fit(X_train, y_train)

"""Si effettui la predizione sui dati di test."""

y_pred = model.predict(X_test)

"""Si determini l'accuratezza del modello."""

accuracy_score(y_test, y_pred)

"""## Conclusioni

Di seguito si illustrino i risultati finali dei vari classificatori addestrati.
"""

print(tabulate([['DecisionTree', accuracy_score(y_test, dt.predict(X_test))], 
                ['K-Nearest Neighbors', accuracy_score(y_test, knn.predict(X_test))],
                ['Neural Network', accuracy_score(y_test, nn.predict(X_test))],
                ['LogisticRegression', accuracy_score(y_test, lgr.predict(X_test))],
                ['Stacking', accuracy_score(y_test, model.predict(X_test))]], headers=['Classifier', 'Accuracy'], tablefmt='orgtbl'))

"""Complessivamente è possibile notare che lo stacking ha migliorato la classificazione rispetto alla classificazione effettuata dai singoli *weak learner*, in particolare:
* lo stacking è migliorato rispetto a `DecisionTree` di circa lo 7.22%;
* lo stacking è migliorato rispetto a `K-Nearest Neighbors` di circa lo 1.11%;
* lo stacking non è migliorato rispetto alla rete neurale;
* lo stacking è migliorato rispetto a `LogisticRegression` di circa lo 1.11%.

Di seguito si illustri un esempio di predizione.
"""

digit = 9
digit_to_predict = dataset.data[digit]
digit_target = dataset.target[digit]

digit_predicted = model.predict([digit_to_predict])

print("Predicted: ", digit_predicted[0])
print("Real: ", digit_target)

digit_image = digit_to_predict.reshape(8, 8)

plt.imshow(digit_image, cmap=plt.cm.gray_r, interpolation='nearest')
plt.axis("off")
plt.show()