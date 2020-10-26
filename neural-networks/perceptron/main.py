import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
from single_perceptron import SinglePerceptron

plt.style.use('seaborn-whitegrid')

# Load the dataset
dataset = load_wine()
x = dataset.data

# Show few lines of the dataset
display(pd.DataFrame(data=dataset.data, columns=dataset.feature_names).head())

# Get the classification targets
y = [1 if y == 1 else -1 for y in dataset.target]

# Split the dataset in train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Setup a vector of iteration numbers
iteration_numbers = np.append(10, np.append(np.random.randint(low=20, high=(len(x_train) - 10), size=9), len(x_train)))
iteration_numbers.sort()
iteration_numbers = np.unique(iteration_numbers)

learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]

# Create a set of models for each learning rate
for learning_rate in learning_rates:

    predictions = []
    train_scores = []
    test_scores = []

    # Create a model for each iteration number
    for num in iteration_numbers:
        model = SinglePerceptron(learning_rate=learning_rate, seed=23)
        model.fit(x_train, y_train, iteration_number=num)

        predictions.append(model.predict(x_test))

        train_scores.append(model.score(x_train, y_train))
        test_scores.append(model.score(x_test, y_test))

    # Show table
    columns = ('No. iterations', 'Train score', 'Test score')

    cell_text = []
    for row in range(0, len(iteration_numbers)):
        cell_text.append([iteration_numbers[row], train_scores[row], test_scores[row]])

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    plt.table(cellText=cell_text,
              colLabels=columns,
              loc='center')

    plt.title("Learning rate = {}".format(learning_rate))
    plt.show()

    # Show charts
    test = []
    for i in range(0, len(x_test)):
        test.append(i)

    # Find the index of the best model
    index = int(np.argmax(test_scores))
    print("Best model: index = {}, score = {}".format(index, np.max(test_scores)))

    x_wrong_prediction = []
    y_wrong_prediction = []
    y_correct_prediction = []

    for i in range(0, len(predictions[index])):
        if predictions[index][i] != y_test[i]:
            x_wrong_prediction.append(i)
            y_wrong_prediction.append(predictions[index][i])
            y_correct_prediction.append(y_test[i])

    plt.style.use('seaborn-whitegrid')

    plt.title("Classification chart")
    plt.xlabel("Values")
    plt.ylabel("Classification")
    plt.scatter(test, predictions[index], label="Prediction")
    plt.scatter(test, y_test, label="Original values")
    plt.legend()
    plt.show()

    plt.title("Wrong predictions chart")
    plt.xlabel("Values")
    plt.ylabel("Classification")
    plt.scatter(x_wrong_prediction, y_wrong_prediction, label="Wrong predictions")
    plt.scatter(x_wrong_prediction, y_correct_prediction, label="Y test")
    plt.legend()
    plt.show()

    plt.title("Scores")
    plt.xlabel("Iteration number")
    plt.ylabel("Score")
    plt.plot(iteration_numbers, train_scores, label="Train score")
    plt.plot(iteration_numbers, test_scores, label="Test score")
    plt.legend()
    plt.show()

# sklearn
model = Perceptron(tol=1e-3, random_state=23, max_iter=len(x))
model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print("Train score: {}".format(train_score))
print("Test score: {}".format(test_score))
