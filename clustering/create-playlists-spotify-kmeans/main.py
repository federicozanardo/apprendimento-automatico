# -*- coding: utf-8 -*-
"""Creare delle playlist per Spotify con K-Means.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a2zcEjYd8pFCFJUGCqv1M_7ItNw6FqMW

# **Creare delle playlist per Spotify con K-Means**

## Introduzione

In questo notebook ho utilizzato un algoritmo di clustering per creare delle playlist di Spotify. In particolare ho utilizzato l'algotimo partizionale *K-Means*. Per una spiegazione più completa riguardo gli algortimi di clustering e l'algoritmo K-Means si può far riferimento al seguente [notebook](https://colab.research.google.com/drive/1Eje2F3pvMOL8ukVKtIoeVaDPaKILfEjV).

### **Dataset**

Il dataset è reperibile su [Kaggle](https://www.kaggle.com/tomigelo/spotify-audio-features). In particolare è stato utilizzato `SpotifyAudioFeaturesApril2019.csv`.

## Caricamento del dataset e delle librerie necessarie
"""

import numpy as np
import pandas as pd

# Preprocessing
from sklearn.preprocessing import MinMaxScaler

# Clustering algorithms
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy

# Metrics
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

import matplotlib.pyplot as plt
import seaborn as sns

!pip install tqdm
from tqdm import tqdm
from tabulate import tabulate

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('SpotifyAudioFeaturesApril2019.csv')

"""## Esplorazione del dataset

Si illustrino le feature presenti nel dataset:
* `artist_name`: specifica il nome dell'artista;
* `track_id`: identificativo della traccia Spotify;
* `track_name`: nome della traccia;
* `acousticness`: indica misura di confidenza compresa tra 0 e 1 del fatto che la traccia sia *acustica*. 1 rappresenta un'elevata confidenza che la traccia sia acustica. Una traccia acustica usa solamente, o principalmente, strumenti che producono il suono con mezzi acustici, anziché elettrici o elettronici;
* `danceability`: indica quanto sia adatta una traccia per ballare sulla base di una combinazione di elementi musicali tra cui il tempo, la stabilità del ritmo, la forza del battito e la regolarità generale. 0 siginifica che è il meno ballabile e 1 che è il più ballabile;
* `duration_ms`: la durata della traccia in millisecondi;
* `energy`: indica una misura compresa tra 0 e 1 e rappresenta una misura percettiva di intensità e attività. In genere, le tracce energiche sono veloci e rumorose. Le caratteristiche percettive che contribuiscono a questo attributo includono la gamma dinamica, il volume percepito, il timbro, la frequenza di inizio e l'entropia generale;
* `instrumentalness`: indica se una traccia non contiene voci. Più il valore di strumentalità è vicino a 1, maggiore è la probabilità che la traccia non contenga contenuti vocali. I valori superiori a 0.5 rappresentano tracce strumentali, ma la fiducia è maggiore quando il valore si avvicina a 1;
* `key`: la chiave complessiva stimata della traccia;
* `liveness`: indica la presenza di un pubblico nella registrazione. Valori di vivacità più elevati rappresentano una maggiore probabilità che la traccia sia stata eseguita dal vivo. Un valore superiore a 0.8 fornisce una forte probabilità che la traccia sia live;
* `loudness`: indica il volume complessivo di una traccia in *decibel* (*dB*). I valori della sonorità vengono mediati su tutta la traccia e sono utili per confrontare il volume relativo delle tracce. I valori tipici sono compresi tra -60 e 0 db;
* `mode`: indica la modalità (*maggiore* o *minore*) di una traccia, il tipo di scala da cui deriva il suo contenuto melodico. Maggiore è rappresentato da 1 e minore è 0;
* `speechiness`: indica la presenza di parole pronunciate in una traccia. Più la registrazione è simile ad un discorso (ad esempio talk show, audiolibro, poesia), più è vicino a 1. I valori superiori a 0.66 descrivono tracce che probabilmente sono composte interamente da parole pronunciate. I valori compresi tra 0.33 e 0.66 descrivono brani che possono contenere sia musica che parlato. I valori inferiori a 0.33 molto probabilmente rappresentano musica e altre tracce non simili al parlato;
* `tempo`: indica il tempo stimato complessivo di una traccia in *battiti al minuto* (*BPM*). Il tempo è la velocità o il ritmo di un dato brano e deriva direttamente dalla durata media del battito;
* `time_signature`: è una convenzione notazionale per specificare quanti battiti ci sono in ogni misura;
* `valence`: indica una misura da 0 a 1 che descrive la positività musicale trasmessa da una traccia. I brani con valenza alta suonano più positivi (ad esempio: felice, allegro, euforico), mentre i brani con valenza bassa suonano più negativi (ad esempio: triste, depresso, arrabbiato);
* `popularity`: indica la popolarità della traccia.
"""

df.head()

df.info()

df.describe()

"""Si considerino soltanto le canzoni che hanno una popolarità maggiore di 70."""

df = df[df.popularity > 70]

df.shape

"""Si osservino graficamente le correlazioni presenti tra le varie feature."""

df_corr = df.corr(method='pearson')

fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df_corr, annot=True, cbar=True)
plt.show()

"""## Preparazione del dataset e preprocessing

Si proceda nel rimuovere alcune feature.
"""

X = pd.DataFrame(df.drop(['artist_name', 'track_id', 'track_name', 'track_id', 'duration_ms', 'key', 'mode', 'tempo', 'time_signature', 'valence'], axis=1).values)

cols = df.drop(['artist_name', 'track_id', 'track_name', 'track_id', 'duration_ms', 'key', 'mode', 'tempo', 'time_signature', 'valence'], axis=1).columns
X.columns = cols

X.head()

"""Si effettui lo scaling delle feature con `MinMaxScaler`."""

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X))
X_scaled.columns = cols

X_scaled

X_scaled = scaler.fit_transform(X)

"""## Realizzazione del modello

### **Elbow method**

Quando si vuole utilizzare K-Means bisogna impostare un numero di cluster in cui suddividere i dati. Poichè non si dispone di alcuna conoscenza a priori, la scelta del numero di cluster è arbitrario. Per trovare un valore preciso che possa fornire delle buone prestazioni si utilizza l'*elbow method*. Questa tecnica consiste nell'eseguire l'algortimo K-Means con un diverso numero di cluster fissati. Le *inerzie* (*WCSS*) ottenute per ogni iterazione vengono memorizzate e tracciate su un grafico. A questo punto si determina l'*elbow* nel grafico, quindi, la posizione in cui la linea inizia ad appiattirsi, facendola sembrare un gomito.

### **Inerzia**

L'algoritmo K-Means cerca di scegliere i centroidi che minimizzano l'*inerzia*, o il *criterio della somma dei quadrati* (*WCSS*, *within-cluster sum-of-squares criterion*) all'interno del cluster:
\begin{equation}
\sum_{i = 0}^n min_{\mu_j \in C} (||x_i - \mu_j||^2)
\end{equation}

L'inerzia può essere intesa come una misura di quanto siano *internamente coerenti* i cluster.

Si esegue l'algoritmo K-Means con un numero di differente di cluster e si memorizzino le varie inerzie.
"""

wcss = []

for i in tqdm(range(1, 21)):
  kmeans = KMeans(n_clusters=i, max_iter=1000, random_state=42, n_jobs=-1, precompute_distances=True)
  kmeans.fit(X_scaled)
  wcss.append(kmeans.inertia_)

"""Si illustri in forma tabulare e graficamente i risultati ottenuti."""

print(tabulate([[i + 1, wcss[i]] for i in range(0, 20)], 
               headers=['No of cluster', 'WCSS'], 
               tablefmt='orgtbl'))

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(range(1, 21), wcss, c='lightblue')
plt.scatter(range(1, 21), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

"""Si fissi il numero di cluster a 5. A questo punto è possible eseguire l'algoritmo K-Means per realizzare le playlist."""

n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_jobs=-1, precompute_distances=True, max_iter=1000)

y_pred = kmeans.fit_predict(X_scaled)

y_pred

y_pred.shape

"""Si illustrino graficamente i cluster appena realizzati, con i corrispettivi centroidi."""

fig, ax = plt.subplots(figsize=(20, 15))
colors = ['red', 'blue', 'green', 'cyan', 'purple', 'grey', 'lightblue', 'yellow', 'magenta', 'lightgreen', 'tomato', 'indigo', 'lightsalmon']

for i in range(n_clusters):
  plt.scatter(X_scaled[y_pred==i, 0], X_scaled[y_pred==i, 1], s=50, c=colors[i], label ='Cluster {}'.format(i + 1))

for i in range(n_clusters):
  ax.annotate(i + 1, (kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1]))
  plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1], s=300, c='orange', marker='o')

plt.legend()
plt.show()

"""Dal grafico si può notare che i cluster 1 e 5 sono distanziati tra loro e dagli altri cluster, pertanto ipotizzo che l'algoritmo avrà creato delle playlist con canzoni molto diverse tra loro. I cluster 2, 3 e 4 hanno i rispettivi centroidi vicini tra loro, pertanto mi aspetto che avranno alcune similarità tra loro.

## Valutazione dei cluster

### **Analisi preliminari**

In questa sezione verranno esaminati tutti i cluster realizzati.
"""

kmeans = pd.DataFrame(data=y_pred, dtype=int)
kmeans.columns = ['k_cluster']
print(kmeans.shape)
kmeans.head()

df.reset_index(drop=True, inplace=True)
kmeans.reset_index(drop=True, inplace=True)

df_cluster = pd.concat([df, kmeans], axis=1)
print(df_cluster.shape)
df_cluster.head()

df_cluster.dropna()

"""Si raggruppino i vari cluster e si ottengano alcune informazioni su di essi."""

df_cluster.groupby("k_cluster").describe()

"""Si determini il numero di oggetti (canzoni) presenti in ogni cluster (playlist)."""

df_cluster['k_cluster'].value_counts()

"""Si determini la media di ogni feature per ogni cluster."""

columns=['Cluster', 
         'Mean acousticness', 
         'Mean danceability', 
         'Mean energy', 
         'Mean instrumentalness', 
         'Mean liveness', 
         'Mean loudness', 
         'Mean speechiness', 
         'Mean popularity']

data = []
for i in range(0, n_clusters):
  cluster = df_cluster[df_cluster['k_cluster'] == i]
  data.append([(i + 1),
               cluster.acousticness.mean(), 
               cluster.danceability.mean(),
               cluster.energy.mean(),
               cluster.instrumentalness.mean(),
               cluster.liveness.mean(),
               cluster.loudness.mean(),
               cluster.speechiness.mean(),
               cluster.popularity.mean()])

df_means = pd.DataFrame(data=data)
df_means.columns = columns

df_means

df_means = df_means.drop(['Cluster'], axis=1)

"""Si calcolino le correlazioni tra le feature ottenute precedentemente."""

df_means_corr = df_means.corr(method='pearson')
df_means_corr

"""Per ogni feature si analizzino le varie correlazioni. Si considerino soltanto le feature che hanno una correlazione maggiore o uguale a 0.80:
* **Acousticness**: le feature elencate sono tutte inverse rispetto all'*acusticità*. Il motivo è che una canzone acustica non ha la stessa vivacità di una canzone realizzata con degli strumenti elettrici o elettronici. L'acusticità è:
  * inversa rispetto a *danceability*
  * inversa rispetto a *energy*
  * inversa rispetto a *liveness*
  * inversa rispetto a *loudness*

* **Danceability**
  * diretta rispetto a *energy*: considerando i concetti di ballabilità e di energia definiti precedentemente, più una canzone è energica, più è ballabile;
  * inversa rispetto a *instrumentalness*: una canzone instrumental non ha una parte cantata, pertanto risulta essere meno ballabile;
  * diretta rispetto a *liveness*: se la canzone è stata registrata durante un concerto risulterà essere più ballabile;
  * diretta rispetto a *loudness*: più una canzone ha un valore in decibel maggiore, più sarà, in un certo senso, *vivace*;

* **Energy**
  * inversa rispetto a *instrumentalness*: tenendo in considerazione il concetto di energia, una traccia instrumental, che non contiene una parte cantata, sarà meno *energica*;
  * diretta rispetto a *liveness*
  * diretta rispetto a *loudness*

* **Instrumentalness**
  * inversa rispetto a *loudness*

* **Liveness**
  * diretta rispetto a *loudness*

Si definisca una funzione che permetta di illustrare graficamente per ogni feature il corrispondente grafico.
"""

def show_plots(X, n_cluster):
  fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(25, 10))
  sns.distplot(X[X['k_cluster'] == n_cluster].acousticness, ax=ax[0, 0])
  sns.distplot(X[X['k_cluster'] == n_cluster].danceability, ax=ax[0, 1])
  sns.distplot(X[X['k_cluster'] == n_cluster].instrumentalness, ax=ax[0, 2])
  sns.distplot(X[X['k_cluster'] == n_cluster].liveness, ax=ax[0, 3])
  sns.distplot(X[X['k_cluster'] == n_cluster].loudness, ax=ax[1, 0])
  sns.distplot(X[X['k_cluster'] == n_cluster].speechiness, ax=ax[1, 1])
  sns.distplot(X[X['k_cluster'] == n_cluster].popularity, ax=ax[1, 2])
  sns.distplot(X[X['k_cluster'] == n_cluster].energy, ax=ax[1, 3])
  fig.suptitle("Cluster {}".format(n_cluster + 1), fontsize=20)
  plt.show()

"""### **Cluster 1**"""

df_cluster.loc[df_cluster['k_cluster'] == 0].head()

show_plots(df_cluster, 0)

"""Dal grafico è possibile evincere che le canzoni in questa playlist sono principalmente *acustiche* e *instrumentali*. Inoltre:
* *energy* è inversa rispetto a *instrumentalness*;
* *acousticness* è inversa rispetto a *danceability*;
* *acousticness* è inversa rispetto a *loudness*.

Quindi, le canzoni presenti in questa playlist saranno delle canzoni *tranquille*.

### **Cluster 2**

I cluster 2, 3 e 4 sono simili tra loro. Pertanto si cercherà di fare un'analisi più precisa sui vari oggetti contenuti in essi.
"""

df_cluster[df_cluster['k_cluster'] == 1].head(15)

show_plots(df_cluster, 1)

"""Le canzoni in questa playlist sono *energiche* e hanno un valore medio alto di *loudness* (hanno il più alto valore di energy e loudness). Conseguenza del fatto di avere un alto valore di *energy* è la bassa *acusticità* delle canzoni. Queste conclusioni rispettano le analisi effettuate precedentemente (si veda la seguente [sezione](https://colab.research.google.com/drive/1a2zcEjYd8pFCFJUGCqv1M_7ItNw6FqMW#scrollTo=9-cyJJ0yJg26)).

È possibile notare un'"anomalia": sulla base delle considerazioni fatte precedentemente, se un cluster ha un alto valore di *energy* dovrebbe anche avere un alto valore di *danceability*. Tuttavia, è possibile notare che in questo cluster non accade tutto ciò. Probabilmente le tracce che costituiscono questa playlist sono energiche ma non si "prestano" ad essere ballate.

### **Cluster 3**
"""

df_cluster[df_cluster['k_cluster'] == 2].head(15)

show_plots(df_cluster, 2)

"""I brani che appartengono a questo cluster sono *ballabili* e hanno una *speechiness* alta. Infatti, questo cluster ha i valori più alti di danceability e speechiness. Conseguenza di tutto ciò, è il fatto di avere il valore più basso di *instrumentalness*.

Questa playlist sembra essere, in un certo senso, *opposto* alla playlist 2. Infatti, l'*anomalia* in questo caso è il basso valore medio di *energy*. Quest'ultimo valore dovrebbe essere alto grazie all'alto valore di ballabilità (sempre sulla base delle considerazioni fatte preliminarmente alla seguente [sezione](https://colab.research.google.com/drive/1a2zcEjYd8pFCFJUGCqv1M_7ItNw6FqMW#scrollTo=9-cyJJ0yJg26)). Tuttavia si può notare che tutto ciò non accade.

### **Cluster 4**
"""

df_cluster[df_cluster['k_cluster'] == 3].head(15)

show_plots(df_cluster, 3)

"""Si può affermare che questa playlist è simile alle playlist 2 e 3. Tuttavia, è possibile notare che la *popolarità* delle tracce che costituiscono questa playlist è maggiore rispetto a tutte le altre. È possibile inoltre notare che le canzoni hanno una alta *danceability* ed *energy*. Sulla base anche delle considerazione effettuate precedentemente, una conseguenza dell'alta danceability ed energy è la bassa *instrumentalness*.

### **Cluster 5**
"""

df_cluster[df_cluster['k_cluster'] == 4].head()

show_plots(df_cluster, 4)

"""In questo cluster sono presenti canzoni *ballabili* ma con una certa *acusticità*. Prendendo casualmente qualche canzone da questa playlist si può notare che diverse tracce sono state realizzate con strumenti acustici e sono *ritmiche*, il che aumenta il fattore di *danceability* della canzone.

## Conclusioni

Riassumendo, l'algoritmo di clustering K-Means ha prodotto le seguenti playlist:
* **Cluster 1**: è una playlist di canzoni *tranquille* / *rilassanti*;
* **Cluster 2**: contiene brani *energici* e *vivaci*, tuttavia via non adatti ad essere *ballati*;
* **Cluster 3**: contiene dell tracce che possono essere *ballate* e che hanno una *speechiness* più alta rispetto alle altre playlist;
* **Cluster 4**: è una playlist di tracce *popolari* che sono *energiche*  e *ballabili*;
* **Cluster 5**: è una playlist adatta a chi preferisce delle canzoni realizzate con strumenti *acustici*, ma che allo stesso tempo hanno anche un certo *ritmo*.

Il modello non ha prodotto dei cluster con degli oggetti *profondamente* differenti tra loro. Soltanto il cluster 1 e il cluster 5 contengono degli oggetti molto differenti tra loro e tra gli oggetti degli altri cluster. Gli oggetti dei cluster 2, 3 e 4 sono simili tra loro anche se hanno alcune lievi differenze. Tali playlist potrebbero soddisfare i gusti degli ascoltatori che ricercano delle playlist più particolari o di nicchia.
"""