# -*- coding: utf-8 -*-
"""SVM per Credit Card Fraud Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vBZG4pvVkC52UIwZBPY9OIwn-48oOvv6

# SVM per Credit Card Fraud Detection

## Dataset

Il dataset contiene le transazioni effettuate con carte di credito nel settembre 2013 da titolari di carta europei. Questo dataset presenta le transazioni avvenute in due giorni, dove sono state registrate 492 frodi su 284 807 transazioni. Il dataset è altamente sbilanciato: la classe positiva (le frodi) rappresenta lo 0.172% di tutte le transazioni.

Il dataset contiene soltanto feature numeriche che sono il risultato di una trasformazione PCA. Per questioni di privacy, il dataset non provvede le feature originali ed ulteriori informazioni di base sui dati. Le feature `V1`, `V2`, … `V28` sono le componenti principali ottenute tramite la PCA. Le uniche feature che non sono state trasformate tramite la PCA sono `Time` e `Amount`.

La feature `Time` contiene i secondi trascorsi tra ogni transazione e la prima transazione nel datatset. La feature `Class` è il target e assume valore 1 in caso di frode e 0 in caso contrario.

Il dataset è reperibile su [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Upload ed esplorazione del dataset

Caricamento delle librerie necessarie.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

"""Upload del dataset."""

dataset = pd.read_csv('creditcard.csv')
df = pd.DataFrame(dataset)

"""Si osservino le prime righe del dataset."""

df.head()

"""Si ottengano delle informazioni più precise riguardo le informazioni contenute nel dataset. In particolare, si osservano i seguenti indicatori:
* Sum
* Average
* Variance
* Minimum
* 1st quartile
* 2nd quartile
* 3rd quartile
* Maximum
"""

df.describe()

"""Si calcolino le correlazioni tra le feature a coppie."""

df_corr = df.corr()

"""Si illustrino graficamente tali correlazioni tramite una *heatmap*."""

plt.figure(figsize=(15,10))
sns.heatmap(df_corr)
sns.set(font_scale=1, style='white')

plt.title('Correlazioni')
plt.show()

"""## Preparazione del dataset

Si osservi la quantità di transazioni classificate come frodi e come non frodi.
"""

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
legend = ['frauds', 'non-frauds',]
data = [(df["Class"] == 0).sum(), (df["Class"] == 1).sum()]
ax.bar(legend, data)
plt.show()

"""È chiaro che la classe che costituisce l'insieme delle frodi nelle transazioni è *sottorappresentata* (0,17% dell'intero dataset). Addestrando il modello su questo dataset, esso sarà inefficiente e sarà in grado di prevedere soltanto i casi in cui non sono avvenute delle frodi nelle transazioni. Se il modello venisse testato su questo dataset è possibile che si ottenga un'elevata precisione. Tuttavia, siccome il dataset non è bilanciato, bisogna tener conto della *recall*.

Per risolvere questo problema, si attua un *sottocampionamento* (*under-sampling*). Il sottocampionamento del dataset comporta il mantenimento dei dati sottorappresentati (le transazioni classificate come frodi) aggiungendo lo stesso numero di caratteristiche con `Class = 0` (le transazioni classificate come non frodi) per creare un nuovo dataset che comprende una rappresentazione uguale per entrambe le classi.
"""

fraud_index = np.array(df[df["Class"] == 1].index)

# Take the list of normal indexes from the original dataset
normal_index= df[df["Class"] == 0].index

No_of_frauds= len(df[df["Class"] == 1])
No_of_normals = len(df[df["Class"] == 0])

# Choose indexes at random
# The number of normal transactions must be equal to the number of fraudulent transactions
random_normal_indices= np.random.choice(normal_index, No_of_frauds, replace=False)
random_normal_indices= np.array(random_normal_indices)

# Concatenate fraud indexes and normal indexes to create a single list of indexes
undersampled_indices= np.concatenate([fraud_index, random_normal_indices])

# Use the undersampled indexes to build the a new dataframe
undersampled_data= df.iloc[undersampled_indices, :]

print(undersampled_data.head())

"""Si applichi lo `StandardScaler` alla feature `Amount`.


"""

scaler = StandardScaler()
undersampled_data["Amount"] = scaler.fit_transform(undersampled_data.iloc[:,29].values.reshape(-1,1))

"""Si rimuova la feature `Time` dal dataset."""

undersampled_data= undersampled_data.drop(["Time"], axis= 1)

X = undersampled_data.iloc[:, undersampled_data.columns != "Class"].values

y = undersampled_data.iloc[:, undersampled_data.columns == "Class"].values

"""## Model selection

Si utilizzi la *K-Fold Cross Validation* per determinare i parametri che permettono di ottenere una SVM performante sul dataset. I parametri in questione sono:
* per i kernel **lineare** e **rbf**:
  * `C`: è un parametro di regolarizzazione che permette di controllare l'overfitting. Più alto è il valore di C, minore è la tolleranza e ciò che viene addestrato è un classificatore a margine massimo. Più piccolo è il valore di C, maggiore è la tolleranza di un'errata classificazione e ciò che viene addestrato è un classificatore che generalizza meglio del classificatore a margine massimo. Il valore C controlla la penalità di una classificazione errata. Un valore elevato di C comporterebbe una penalità maggiore per una classificazione errata e un valore inferiore di C comporterebbe una minore penalità per una classificazione errata. Una valore inferiore per questo parametro indurrà ad un margine più ampio, quindi una funzione decisionale più semplice, a scapito della precisione dell'addestramento;
  * `gamma`: definisce quanto lontano arriva l'influenza di un singolo esempio di training, con valori bassi che significano "lontano" e valori alti che significano "vicino". I valori più bassi di gamma danno come risultato modelli con una precisione inferiore o uguale ai valori più alti di gamma;
* per il kernel **poly**, oltre ai parametri esplicitati per i kernel precedenti, si definiscono:
  * `degree`: definisce il grado del polinomio;
  * `coefficient`: controlla quanto il modello è influenzato da polinomi di grado alto. È un termine indipendente dalla funzione kernel.

Definisco dei valori per `C`.
"""

C_range = [2**i for i in range(-5, 5)]
C_range

"""Definisco dei valori per `gamma`.


"""

g_range = [10**i for i in range(-4, 4)]
g_range

"""Definisco dei vari gradi di polinomio."""

degrees = [2 + i for i in range(4)]
degrees

"""Definisco dei valori per `coef0`."""

coefficients = [2**i for i in range(-2, 2)]
coefficients

"""Creo un dizionario di dizionari che contiene la griglia dei parametri per SVM."""

p_grid = {
    "svm" : [
              {"C": C_range, "kernel": ["linear", "rbf"], "gamma" : g_range},
			        {"C": C_range, "kernel": ["poly"], "gamma" : g_range, "degree": degrees, "coef0": coefficients}
            ]
}
p_grid

"""Si definisca 5-Fold Cross Validation."""

skf = KFold(n_splits=5, shuffle=False, random_state=42)

fold, accs = 1, []

for train, test in skf.split(X, y):
  print("FOLD:", fold)
  X_train, X_test = X[train], X[test]

  clf = GridSearchCV(SVC(), param_grid=p_grid["svm"], cv=5, scoring='accuracy')
  clf.fit(X_train, y[train].ravel())

  print("VALIDATION score:", clf.best_score_)
  print("BEST parameters:", clf.best_params_)

  y_pred = clf.predict(X_test)
  y_test = y[test]

  print(classification_report(y_test, y_pred))
  print(confusion_matrix(y_test, y_pred))

  acc = accuracy_score(y_test, y_pred)
  print("TEST score:", str(acc))
  print()

  accs.append(acc)

  fold += 1

print("AVG ACCURACY", np.mean(accs), "+-", np.std(accs))

"""Si definisca una funzione che permette di ottenere alcuni valori per valutare il modello."""

def evaluate(y_test, y_pred):
  print("accuracy:", accuracy_score(y_test, y_pred))
  print("precision:", precision_score(y_test, y_pred))
  print("recall:", recall_score(y_test, y_pred))

"""Si suddivida il dataset in training set e test set."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

"""Si definisca il modello con i parametri determinati durante la model selection."""

classifier = SVC(C=2, gamma=0.001, kernel='poly', degree=5, coef0=2)
classifier.fit(X_train, y_train.ravel())

"""Si effettui una predizione sul test set."""

y_pred = classifier.predict(X_test)

"""Si mostrino le valutazioni del modello appena realizzato."""

evaluate(y_test, y_pred)

"""Si mostri inoltre la *confusion matrix* del modello."""

confusion_matrix(y_test, y_pred)