# Preprocessing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qXRSQZDgTrknjUBVj9r31kRFNhNlMYgl)

In this notebook I want to highlight the importance of the dataset preprocessing phase. 
In particular, starting from a dataset, we want to:
1. analyze it to understand which features are relevant for classification and which ones are irrelevant for prediction;
2. analyze feature values.

The dataset used in this notebook concerns the passengers of the Titanic ocean liner. 
The purpose of the classification is to predict, based on a set of features, whether passengers will survive or not.

I used an SVM to do the classification. The purpose of this notebook is not to find the ideal algorithm or the ideal 
parameters for the classification algorithm. Therefore, the algorithm and related parameters chosen may not be ideal 
for making an excellent prediction. To determine the same conclusions, different algorithms and parameters can be used.

## Dataset

The dataset is provided by [Kaggle](https://www.kaggle.com/c/titanic/data).
