from sklearn.linear_model import Perceptron

neuron = Perceptron()

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 0, 0, 1]

neuron.fit(X, Y)
print(neuron.score(X, Y))
print(neuron.predict(X))
