"""
Notebook: https://colab.research.google.com/drive/1qXRSQZDgTrknjUBVj9r31kRFNhNlMYgl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('./dataset/train.csv')

# Explore the dataset
print(df.head())
print(df.info())
print(df.shape)

# Analyze the features

# Sex
sns.countplot(x='Sex', hue='Survived', data=df)
plt.show()

# Number of family members of the passenger
sns.countplot(x='SibSp', hue='Survived', data=df)
plt.show()

# Number of parents/children
sns.countplot(x='Parch', hue='Survived', data=df)
plt.show()

# Ticket class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.show()

# Analyze how the age of passengers is distributed.
sns.distplot(df.Age)
plt.show()

# Get the median
print("Median of the age: {}".format(df.Age.median()))

# Plot the distribution of the age
sns.countplot(x='Age', hue='Survived', data=df)
plt.show()

# Fare for the passenger. This feature seems not to have a correlation useful for the classification
sns.countplot(x='Fare', hue='Survived', data=df)
plt.show()

# Preprocessing

# Feature selection: I remove from the dataset the features that I consider not useful for classification

X = df.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Survived', 'PassengerId'], axis=1)
print(X)

y = df['Survived']

# Encoding of feature values

# Check that the features doesn't have missing values
print("Pclass: {}".format(X.Pclass.isnull().sum()))
print("Sex: {}".format(X.Sex.isnull().sum()))
print("Age: {}".format(X.Age.isnull().sum()))
print("SibSp: {}".format(X.SibSp.isnull().sum()))
print("Parch: {}".format(X.Parch.isnull().sum()))
print("Embarked: {}".format(X.Embarked.isnull().sum()))

# Encode the values of feature 'Sex' with 'OneHotEncoder'
X['Sex'] = OneHotEncoder(categories='auto').fit_transform(np.array(X['Sex'].values)[:, np.newaxis]).toarray()

# Regarding the `Age` feature, I have previously noticed that there are some lines that have no value.
# To solve this problem, I decide to fill these values with the median found in the exploration phase of the dataset.
# Another method that can be applied is to use the most frequent value in the distribution.
X.Age.fillna(X.Age.median(), inplace=True)

# In the case of the `Embarked` feature I decide to fill the missing values with the most frequent value in that feature
X['Embarked'] = SimpleImputer(missing_values=np.NaN, strategy='most_frequent').fit_transform(
    np.array(X['Embarked'].values)[:, np.newaxis])

# Apply the 'LabelEncoder'
label_encoder = LabelEncoder()
X['Embarked'] = label_encoder.fit_transform(X['Embarked'])

print("Classes")
print(label_encoder.classes_)

# Split the dataset in training set and test set
X_train, X_test, y_train, y_test1 = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature scaling

# Observe how the data are distributed before doing the scaling
sns.distplot(X_train.Age)
plt.show()

# Apply the 'StandarScaler' to the values of 'Age'
scaler = StandardScaler()
X_train.Age = scaler.fit_transform(X_train.Age.values.reshape(-1, 1))

print(X_train.Age.head())

# Plot the distribution after the scaling
sns.distplot(X_train.Age)
plt.show()


# Classification

# Use the preprocessed dataset. I use a SVM to do the classification. The purpose of this project is not to find the
# best prediction algorithm or the best parameters

# Print some evaluations
def evaluate(y_test, y_pred):
    print("accuracy:", accuracy_score(y_test, y_pred))
    print("precision:", precision_score(y_test, y_pred))
    print("recall:", recall_score(y_test, y_pred))


# Setup the SVM
svc = SVC(kernel="rbf", gamma=0.1)

# Train the algorithm
svc.fit(X_train, y_train)

# Scale X_test.Age
X_test.Age = scaler.transform(X_test.Age.values.reshape(-1, 1))

# Prediction
y_pred1 = svc.predict(X_test)

# Evaluation
evaluate(y_test1, y_pred1)

# Add a feature

# To see if the previously implemented feature selection has led to any benefits, I create another dataset containing
# of features not relevant for classification purposes. A clarification: the addition of non-relevant features
# they can create a degradation of the score of the algorithm used. The experiment is carried out in parts of the
# algorithm of classification used (including related parameters).

# In this new dataset I add the feature 'Cabin'
X_dirty = df.drop(['Name', 'Ticket', 'Fare', 'Survived', 'PassengerId'], axis=1)
print(X_dirty)

# Re-do the preprocessing of the previous features

# Sex
X_dirty['Sex'] = OneHotEncoder(categories='auto').fit_transform(
    np.array(X_dirty['Sex'].values)[:, np.newaxis]).toarray()

# Age
X_dirty.Age.fillna(X_dirty.Age.median(), inplace=True)

# Embarked
X_dirty['Embarked'] = SimpleImputer(missing_values=np.NaN, strategy='most_frequent').fit_transform(
    np.array(X_dirty['Embarked'].values)[:, np.newaxis])

label_encoder = LabelEncoder()
X_dirty['Embarked'] = label_encoder.fit_transform(X['Embarked'])

# Check that 'Cabin' has no missing values
print("Cabin: {}".format(X_dirty.Cabin.isnull().sum()))

# Fill the missing values with the most present value in the feature
X_dirty['Cabin'] = SimpleImputer(missing_values=np.NaN, strategy='most_frequent').fit_transform(
    np.array(X_dirty['Cabin'].values)[:, np.newaxis])

# Apply the 'LabelEncoder'
label_encoder = LabelEncoder()
X_dirty['Cabin'] = label_encoder.fit_transform(X_dirty['Cabin'])

print("Classes")
print(label_encoder.classes_)
print()

print("X_dirty['Cabin'].head()")
X_dirty['Cabin'].head()

# Split the dataset
X_train, X_test, y_train, y_test2 = train_test_split(X_dirty, y, test_size=0.20, random_state=42)

# Apply the 'StandardScaler'
# Age - scaling
scaler = StandardScaler()
X_train.Age = scaler.fit_transform(X_train.Age.values.reshape(-1, 1))

# Setup the SVM
svc = SVC(kernel="rbf", gamma=0.1)

# Train the algorithm
svc.fit(X_train, y_train)

# Scale X_test.Age
X_test.Age = scaler.transform(X_test.Age.values.reshape(-1, 1))

# Prediction
y_pred2 = svc.predict(X_test)

# Evaluation
print("Evaluation")
evaluate(y_test1, y_pred1)
print()

print("Evaluation with \'Cabin\' feature")
evaluate(y_test2, y_pred2)

# Remove a feature

# A similar argument can also be made if you are going to remove a feature that is important for the classification.

# In this new dataset I remove the 'Age' featue
X_dirty = df.drop(['Age', 'Name', 'Ticket', 'Fare', 'Cabin', 'Survived', 'PassengerId'], axis=1)
print(X_dirty)

# Re-do the preprocessing of the previous features

# Sex
X_dirty['Sex'] = OneHotEncoder(categories='auto').fit_transform(
    np.array(X_dirty['Sex'].values)[:, np.newaxis]).toarray()

# Embarked
X_dirty['Embarked'] = SimpleImputer(missing_values=np.NaN, strategy='most_frequent').fit_transform(
    np.array(X_dirty['Embarked'].values)[:, np.newaxis])

label_encoder = LabelEncoder()
X_dirty['Embarked'] = label_encoder.fit_transform(X['Embarked'])

# Split the dataset
X_train, X_test, y_train, y_test3 = train_test_split(X_dirty, y, test_size=0.20, random_state=42)

# Fit and prediction
svc = SVC(kernel="rbf", gamma=0.1)
svc.fit(X_train, y_train)
y_pred3 = svc.predict(X_test)

# Evaluation
print("Evaluation")
evaluate(y_test1, y_pred1)
print()

print("Evaluation with \'Cabin\' feature")
evaluate(y_test2, y_pred2)
print()

print("Evaluation without \'Age\' feature")
evaluate(y_test3, y_pred3)
