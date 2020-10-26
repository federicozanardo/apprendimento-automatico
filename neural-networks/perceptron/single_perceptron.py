import numpy as np


class SinglePerceptron:

    def __init__(self, learning_rate, seed):
        # Setup the seed for the random generator
        np.random.seed(seed)
        self.learning_rate = learning_rate

    def fit(self, x, t, iteration_number=10):

        # Create randomly the weights vector
        self.w = np.random.random(size=x.shape[1] + 1)

        w_prime = np.zeros(len(self.w))
        score_prime = 0

        for i in range(iteration_number):
            x_prime = np.append([1], x[i])

            # Step function (hard threshold)
            o = np.sign(np.dot(self.w, x_prime))

            # If the classification is wrong, update the weights vector
            if o != t[i]:
                self.w = self.w + np.dot(self.learning_rate * (t[i] - o), x_prime)

            # Determine the score
            score = self.score(x, t)

            # Save the best weights vector and the best score
            if score > score_prime:
                score_prime = score
                w_prime = self.w

        self.w = w_prime

    # Determine the score
    def score(self, x, y):
        correct_prediction_counter = 0

        for i, x_prime in enumerate(x):
            x_prime = np.append([1], x_prime)

            # Check if the model predicted the correct value
            if np.sign(np.dot(self.w, x_prime)) == y[i]:
                correct_prediction_counter += 1

        return correct_prediction_counter / len(y)

    def predict(self, x):
        y = []

        for prediction in x:
            prediction = np.append([1], prediction)
            y.append(np.sign(np.dot(self.w, prediction)))

        return np.array(y)
