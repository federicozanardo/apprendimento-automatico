# -*- coding: utf-8 -*-
"""Autoencoder - Rappresentazione dei dati nello strato nascosto.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/131WGUnBgnBDsB0UeJ-xBHdOoKr1nm1VQ

# Autoencoder - Rappresentazione dei dati nello strato nascosto

## Introduzione

### Obiettivo

L'obiettivo di questo notebook è quello di realizzare una rete neurale multistrato che prende in input una serie di stringhe binarie, costituite da 8 bit ciascuna, e in output deve restituire i medesimi valori ricevuti in input. Una rete neurale così descritta è un **autoencoder**. Gli autoencoder vengono utilizzati per diverse applicazioni come la cancellazione del rumore da un insieme di dati o il rilevamento di anomalie. Lo scopo è quello di osservare in che modo i valori di input vengono rappresentati negli strati nascosti della rete neurale. È di particolare interesse perchè gli strati nascosti della rete possiedono un numero inferiore di nodi, per cui la rete deve trovare una rappresentazione efficace per adempiere allo scopo descritto.

### Descrizione della rete neurale

La rete neurale che si realizzerà avrà 3 strati:
* **Input layer**: è lo strato che riceve in input le stringhe binarie di 8 bit ciascuna;
* **Code layer**: è lo strato intermedio ed ha un minor numero di nodi rispetto agli strati di input e di output;
* **Output layer**: è lo strato che resistuisce i valori prodotti dalla rete neurale. In questo strato viene utilizzata la funzione di attivazione *sigmoidea*.

L'autoencoder ha una struttura simmetrica: a partire dall'input layer fino al code layer, il numero di nodi diminuisce; viceversa, a partire dal code layer fino all'output layer, il numero di nodi aumenta.

## Caricamento delle librerie necessarie
"""

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

tf.random.set_seed(42)

"""## Realizzazione del dataset

Il dataset realizzato è costituito da 8 stringhe binarie di 8 bit ciascuna, in cui i valori sulla diagonale sono pari a 1, mentre gli altri valori sono uguali a 0.

Il valore *target* per ogni esempio del dataset corrisponde all'esempio stesso, in quanto lo scopo dell'autoencoder è proprio quello di restituire in output gli stessi valori ricevuti in input.
"""

X = np.identity(8, dtype=int)

X

"""## Implementazione della rete neurale

Sono state realizzate due rete neurali che svolgono il medesimo compito. Una rete neurale è stata realizzata utilizzando la libreria [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) e l'altra [Keras](https://keras.io/).

### Implementazione della rete neurale con *sklearn*

Di seguito si descrivono i parametri utilizzati per l'implementazione dell'autoencoder:
* `activation='logistic'`: questo parametro specifica la funzione di attivazione che deve essere utilizzata negli strati della rete;
* `hidden_layer_sizes=(3,)`: questo parametro indica il numero di nodi che costituisce lo strato nascosto della rete. Sulla base dell'immagine della slide 15 della lezione 2020-10-28, il numero di nodi è 3;
* `learning_rate_init=0.1`: questo parametro definisce il tasso di apprendimento che viene applicato alla rete neurale;
* `max_iter=5000`: sulla base della descrizione dell'immagine presente nella slide 15, viene definito il numero massimo di iterazioni che devono essere svolte durante il training della rete;
* `random_state=42`: questo parametro specifica il *seed* per il generatore di numeri casuali. Fissando questo valore è possibile rendere l'esperimento riproducibile e di conseguenza è possibile svolgere più volte lo stesso esperimento, senza la preoccupazione che ogni volta si producano dei risultati diversi;
* `solver='sdg'`: indica l'algoritmo di *discesa del gradiente stocastico*. Questo algoritmo viene utilizzato per l'aggiornamento dei pesi dei vari strati della rete.

La funzione di *loss* utilizzata in questa rete neurale è la *logistic loss*.
"""

autoencoder = MLPClassifier(hidden_layer_sizes=(3,),
                        activation='logistic',
                        learning_rate_init=0.1,
                        max_iter=5000,
                        random_state=42)

"""Si esegua il training dell'autoencoder."""

autoencoder.fit(X, X)

"""Si osservino le rappresentazioni alternative dell'input usufruite dalla rete neurale, durante la fase di training."""

weights = autoencoder.coefs_[0]
weights

"""Si effetui una predizione fornendo in input il medesimo dataset utilizzato nella fase di training."""

x_pred = autoencoder.predict(X)
x_pred

"""Si illustri l'andamento della funzione *loss*."""

mlp_loss = autoencoder.loss_curve_

plt.title('Logistic Loss')
plt.plot(autoencoder.loss_curve_, label='loss')
plt.grid(True)

plt.legend()
plt.show()

"""### Implementazione della rete neurale con Keras

Di seguito si descrivono la struttura dell'autoencoder i relativi parametri utilizzati.

Nel metodo `__init__` vengono definiti i seguenti layer:
* `self.input_layer`: definisce lo strato di input della rete neurale. È possibile notare che il numero di nodi è pari a 8, ovvero, il numero di bit della stringa di input;
* `self.code_layer`: è lo strato intermedio che ha un numero inferiore di nodi rispetto agli altri due strati, in particolare, ha 3 nodi. Inoltre, in questo strato viene utilizzata la funzione di attivazione *lineare*;
* `self.output_layer`: definisce lo strato di output della rete neurale. È possibile notare che il numero di nodi è pari a 8, ovvero, il numero di bit della stringa di input. In questo strato viene utilizzata la funzione di attivazione *sigmoidea*.

Nel metodo `call` viene definita la struttura dell'autoencoder.
"""

class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()

    self.input_layer = tf.keras.Sequential([
      Input(shape=(8,)),
    ])

    self.code_layer = tf.keras.Sequential([
      Dense(3, activation='linear')
    ])

    self.output_layer = tf.keras.Sequential([
      Dense(8, activation='sigmoid'),
    ])

  def call(self, x):
    input = self.input_layer(x)
    encoded = self.code_layer(input)
    decoded = self.output_layer(encoded)
    return decoded

autoencoder = Autoencoder()

"""Una volta che il modello è stato definito, viene effettuata una fase di *compilazione* che permette di configurare la funzione *loss* e l'algoritmo per l'aggiornamento dei pesi. In particolare, vengono utilizzati:
* `binary_crossentropy`: è la funzione *loss*;
* `adam`: è un algoritmo di discesa del gradiente stocastico.


"""

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

"""Eseguo il training dell'autoencoder, indicando il numero massimo di iterazioni che devono essere effettuate (5000)."""

history = autoencoder.fit(X, X, epochs=5000)

"""Si osservino le rappresentazioni alternative dell'input usufruite dalla rete neurale, durante la fase di training."""

weights = autoencoder.code_layer.get_weights()
weights

"""Si effettui una predizione fornendo in input il medesimo dataset utilizzato nella fase di training."""

x_pred = autoencoder.predict(X)
x_pred

"""Si illustri l'andamento della funzione *loss*."""

plt.title('Binary Cross Entropy Loss')
plt.plot(history.history['loss'], label='loss')
plt.grid(True)

plt.legend()
plt.show()

"""Confronto tra le funzioni loss degli autoencoder realizzati tramite *sklearn* e *Keras*."""

plt.title('Losses')
plt.plot(history.history['loss'], label='keras')
plt.plot(mlp_loss, label='sklearn')
plt.grid(True)

plt.legend()
plt.show()

"""### Funzione sigmoidea

Una funzione sigmoidea è una funzione *continua* e *derivabile* che ha una derivata prima non negativa. La formula è:

$sigmoid(x) = \frac{1}{1+e^{-t}}$

Nel campo delle reti neurali è molto apprezzata in quanto la funzione soddisfa la seguente proprietà (sia $\sigma$ la funzione sigmoidea):

$\frac{\partial(\sigma(x))}{\partial x} = \sigma(x) \cdot (1 - \sigma(x))$

Questa semplice relazione *polinomiale* fra la derivata e la funzione stessa è molto facile da implementare.

Tale funzione può essere utilizzata al posto della *step function*. In particolare, la funzione sigmoidea non indica soltanto la classe di appartenenza di un valore, ma fornisce anche la probabilità con cui tale elemento appartiene ad una determinata classe. 
Di seguito si illustrino le potenzialità dell'applicazione della funzione sigmoidea alle reti neurali.

Il primo vettore risultante dall'autoencoder dovrebbe essere:

```
[1, 0, 0, 0, 0, 0, 0, 0]
```

Possiamo analizzare la probabilità con cui ogni elemento è stato classificato come 0 o come 1. Osservando il primo elemento del vettore prodotto dall'autoencoder con la funzione di attivazione sigmoidea, possiamo notare che la probabilità che tale cifra sia 1 è di circa il 63%.
"""

x_pred[0][0] * 100

"""Osservando invece il secondo elemento del medesimo vettore, si può vedere che la probabilità che tale cifra sia 1 è del 5.7%."""

x_pred[0][1] * 100

"""Quindi è possibile affermare che la probabilità che tale cifra sia 0 è del 94.26%."""

(1 - x_pred[0][1]) * 100

"""Siano $i, j \in \mathbb{N}$ due indici tale che:
* $i$ indica uno dei vettori che costituisce `x_pred`;
* $j$ indica un valore di un vettore `x_pred`$_i$.

Sia $P(x_{ij} = 0)$ la probabilità che la cifra $x_{ij}$ possa essere uguale a 0 e $P(x_{ij} = 1)$ la probabilità che la cifra $x_{ij}$ possa essere uguale a 1. Di seguito si illustrino le probabilità di ogni elemento $j$ del vettore $i = 0$ di `x_pred`.


"""

for j in range(X.shape[0]):
  print("P(x[0][{}] = 1) = {}".format(j, x_pred[0][j] * 100))
  print("P(x[0][{}] = 0) = {}".format(j, (1 - x_pred[0][j]) * 100))
  print()

"""Mostrando graficamente la funzione sigmoidea e i valori del primo vettore prodotto dall'autoencoder, è possibile notare come sono distribuiti i valori rispetto alla sigmoide."""

x = np.arange(-4, 4, 0.1)
sigmoid = 1 / (1 + np.exp(-x))

plt.plot(x, sigmoid)
plt.xlabel('x')
plt.ylabel('f(x) = sigmoid(x)')
plt.grid(True)

for i in range(X.shape[0]):
  plt.scatter(X[i], x_pred[i], color='orange')

plt.show()

"""Sulla base dei valori restituiti dalla funzione di attivazione sigmoidea, è possibile approssimare i valori a 0 o a 1, producendo così gli stessi valori ricevuti in input."""

x_pred = [[(1 if round(p, 0) > 0 else 0) for p in pred] for pred in x_pred]
x_pred