# -*- coding: utf-8 -*-
"""Recommendation System del dataset Goodreads.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bjqf85-huZ5wC7g17kZWqOzohQQxwego

# **Recommendation System del dataset Goodreads**

## Introduzione

In questo notebook si vuole realizzare un *sistema di raccomandazione* per un dataset di rating di libri. Per realizzarlo verrà utilizzata la libreria [Surprise](http://surpriselib.com/). Questa libreria si basa su [scikit-learn](https://scikit-learn.org/stable/), infatti è molto semplice utilizzare la libreria se si conosce già scikit.

## Dataset

Il dataset utilizzato in questo notebook è reperibile su [Kaggle](https://www.kaggle.com/zygmunt/goodbooks-10k).
"""

!pip install surprise

"""## Caricamento delle librerie necessarie e del dataset"""

import numpy as np
import pandas as pd
import scipy as sp

from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise.model_selection.split import KFold
from surprise.model_selection.search import GridSearchCV

import matplotlib.pyplot as plt

df_ratings = pd.read_csv('ratings.csv')

df_ratings = df_ratings[['user_id', 'book_id', 'rating']] 
df_ratings.columns = ['user_id', 'book_id', 'rating']

"""## Esplorazione del dataset

È possibile notare la caratteristica *long tail* che indica la sparsità della *matrice dei rating*. In particolare:
* pochi utenti hanno fornito feeddback per tanti item;
* tanti utenti hanno fornito feeddback per pochi item.
"""

distribution = df_ratings[['user_id', 'book_id']].groupby(['user_id']).count()

plt.figure(figsize=(15, 10))
plt.hist(distribution['book_id'], bins=range(20, 1500, 10))
plt.gca().set_xlabel("Number of users")
plt.gca().set_ylabel("Number of ratings")
plt.show()

"""Si indichi la sparsità della matrice dei rating."""

users = pd.DataFrame(df_ratings['user_id'])
books = pd.DataFrame(df_ratings['book_id'])

sparsity = 1 - (df_ratings.shape[0] / (users.shape[0] * books.shape[0]))
sparsity

"""Si carichi il dataset."""

reader = Reader() 
dataset = Dataset.load_from_df(df_ratings, reader)

"""## Split del dataset"""

trainset, testset = train_test_split(data=dataset, test_size=0.2, random_state=42)

"""## Model selection

In questo notebook si vanno ad utilizzare dei metodi di *filtering collaborativo*. Questi metodi fanno uso delle interazioni tra utenti ed item: Queste interazioni possono essere prese da uno storico o possono derivare da dei feedback implici/espliciti.

Viene utilizzato l'algortimo **SVD** che implemente la *matrice di fattorizzazione probabilistica*.

### **SVD**
"""

param_grid = {
   "n_epochs": [5, 10],
   "lr_all": [0.002, 0.004],
   "reg_all": [0.4, 0.5] 
}

gridCV = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gridCV.fit(dataset)

gridCV.cv_results

"""Si illustrino delle valutazioni *offline*, ovvero, l'utente non è direttamente coinvolto nella valutazione del sistema di raccomandazione."""

gridCV.best_score

predictor = gridCV.best_estimator['rmse']