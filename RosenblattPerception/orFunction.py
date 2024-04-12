import random

from rosenblattPerception import RosenblattPerception
from rosenblattPerception import ActivationFunctions


neuron = RosenblattPerception(
    [random.random(), random.random(), random.random()],
    ActivationFunctions.heavisideSteupFunction,
)

# Dati per addestramento
# tabella della verità della OR
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_labels = [0, 1, 1, 1]

# Addestramento
print("Start Weight", neuron.weights)
for epoch in range(1000):
    print("epoch:", epoch)
    for i, x in enumerate(x_data):
        label = y_labels[i]
        neuron.learn(x, label, learning_rate=0.01)


# verifica dell'addestramento
print(neuron.output([0, 0]))
print(neuron.output([0, 1]))
print(neuron.output([1, 0]))
print(neuron.output([1, 1]))
print("Weight: ", neuron.weights)
