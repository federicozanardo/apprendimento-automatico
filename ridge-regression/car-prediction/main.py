import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.style.use('seaborn-whitegrid')

# Load the dataset
dataset = pd.read_csv('/Users/federico/Documents/Università/Padova/Anno accademico 2020:2021/Materie/Primo '
                      'semestre/Apprendimento Automatico/Esercizi/ridge-regression/car-prediction/dataset/car '
                      'data.csv')

# Find correlations using Pearson
correlations = dataset.corr(method='pearson')
print(correlations)

# Show correlations
pairPlot = sns.pairplot(dataset, height=2.5)
plt.show()

# Setup X and Y
selling_price = list(dataset["Selling_Price"])
present_price = list(dataset["Present_Price"])

# Create a custom DataFrame
data = dataset[["Selling_Price", "Present_Price"]]

# Show the correlation between Selling Price and Present Price
plt.xlabel('Selling Price')
plt.ylabel('Present Price')
plt.scatter(selling_price, present_price)
plt.show()

# Divide the dataset in training set and test set
x_train, x_test, y_train, y_test = train_test_split(data[["Selling_Price"]], data[["Present_Price"]], test_size=0.2, random_state=123)

# Setup degrees and alphas
degrees = [1, 2, 3, 4]
alphas = [[0, 5000, 10000], [0, 5000, 1000000], [0, 5000, 1000000], [0, 50000, 1000000]]

models = []
save_models_per_degree = []

mean_squared_errors = []
r_squares = []
best_models = []

legend = []
colors = ['orange', 'black', 'red']

for degree in degrees:
    mean_squared_errors.clear()
    save_models_per_degree.clear()
    legend.clear()
    i = 0
    print('Degree = {}'.format(degree))

    for alpha in alphas[degree - 1]:
        ridge = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge(alpha=alpha))
        ridge.fit(x_train, y_train)
        prediction = ridge.predict(x_test)

        # Determine R^2 for train set and test set
        r_square_train = ridge.score(x_train, y_train)
        r_square_test = ridge.score(x_test, y_test)
        r_squares.append(r_square_test)

        # Determine MSE
        mse = mean_squared_error(y_test, prediction)
        mean_squared_errors.append(mse)

        # Save the details about the model just generated
        save_models_per_degree.append([alpha, r_square_train, r_square_test, mse])

        print('Alpha = {}'.format(alpha))
        print('R^2 for train set: {}'.format(r_square_train))
        print('R^2 for test set: {}'.format(r_square_test))
        print('MSE: {}\n'.format(mse))

        # Prepare the data to plot
        data_test, test_prediction_data = zip(*sorted(zip(x_test["Selling_Price"], prediction), key=lambda x_0: x_0[0]))

        # Plot the results
        plt.plot(data_test, test_prediction_data, c=colors[i])
        legend.append('Alpha = {}'.format(alpha))

        i = i + 1

    models.append([m for m in save_models_per_degree])

    # Determine the best model based on R^2
    j = np.argmax(r_squares)
    best_models.append(r_squares[j])

    # Plot the dataset and show the results
    plt.title('Degree = {}'.format(degree))
    plt.xlabel('Selling Price')
    plt.ylabel('Present Price')
    plt.scatter(selling_price, present_price)
    plt.legend(legend)
    plt.show()

# Find the index of the best model based on R^2
m = np.argmax(best_models)

print('Best model')
for i, degree in enumerate(models, start=1):
    for element in degree:
        if best_models[m] == element[2]:
            print('Degree = {}\nAlpha = {}\nR^2 train = {}\nR^2 test = {}\nMSE = {}'.format(i, element[0], element[1], element[2], element[3]))

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(30, 10))

# Show the charts about scores
alphas = []
r_sq_train = []
r_sq_test = []
for i, degree in enumerate(models, start=1):
    alphas.clear()
    r_sq_train.clear()
    r_sq_test.clear()

    for a in degree:
        alphas.append(a[0])
        r_sq_train.append(a[1])
        r_sq_test.append(a[2])

    plot = axs[0, (i - 1)]
    plot.set_title('Score with degree = {}'.format(i))
    plot.set(xlabel="Alpha", ylabel="Score")
    plot.plot(alphas, r_sq_train, c='red')
    plot.plot(alphas, r_sq_test, c='green')
    plot.legend(['Train', 'Test'])

# Show the charts about MSE
alphas = []
mses = []
for i, degree in enumerate(models, start=1):
    alphas.clear()
    mses.clear()

    for a in degree:
        alphas.append(a[0])
        mses.append(a[3])

    plot = axs[1, (i - 1)]
    plot.set_title('MSE with degree = {}'.format(i))
    plot.set(xlabel="Alpha", ylabel="MSE")
    plot.plot(alphas, mses, c='blue')

plt.show()
