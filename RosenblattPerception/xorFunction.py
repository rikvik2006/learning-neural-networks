import random

from unusedcode.neuralNetwork import NeuralNetwork
from rosenblattPerception import RosenblattPerception
from rosenblattPerception import ActivationFunctions


notNeuron1 = RosenblattPerception(
    [random.random(), random.random()],
    ActivationFunctions.heavisideSteupFunction,
)
notNeuron2 = RosenblattPerception(
    [random.random(), random.random()],
    ActivationFunctions.heavisideSteupFunction,
)
andNeuron1 = RosenblattPerception(
    [random.random(), random.random(), random.random()],
    ActivationFunctions.heavisideSteupFunction,
)
andNeuron2 = RosenblattPerception(
    [random.random(), random.random(), random.random()],
    ActivationFunctions.heavisideSteupFunction,
)
orNeuron = RosenblattPerception(
    [random.random(), random.random(), random.random()],
    ActivationFunctions.heavisideSteupFunction,
)

orAndDataset = [[0, 0], [0, 1], [1, 0], [1, 1]]
notDataset = [[0], [1]]
andLabels = [0, 0, 0, 1]
orLabels = [0, 1, 1, 1]
notLabels = [0, 1, 1, 1]

layers = [
    [(notDataset, notLabels), notNeuron1, notNeuron2],
    [(orAndDataset, andLabels), andNeuron1, andNeuron2],
    [(orAndDataset, orLabels), orNeuron],
]

# Dati per addestramento
# tabella della verità della XOR

# Addestramento

for i, layer in enumerate(layers):
    print(f"🔥 Layer {i}")
    for i in range(len(layer) - 1):
        neuron = layer[i + 1]
        dataset, labels = layer[0]

        for epoch in range(1000):
            print(f"📦 Epoch {epoch}")
            for i, x in enumerate(dataset):
                label = labels[i]
                neuron.learn(x, label, learning_rate=0.01)


def network_ouput(input):
    n1_activation = notNeuron1.output([input[0]])
    n2_activation = notNeuron2.output([input[1]])
    print("[not] a1, a2: ", n1_activation, n2_activation)
    and1_activation = andNeuron1.output([n1_activation, input[1]])
    and2_activation = andNeuron1.output([n2_activation, input[0]])
    print(
        f"[and] a1 x {input[1]}, a2 x {input[0]}: {and1_activation} {and2_activation}"
    )

    orNeuron_activation = orNeuron.output([and1_activation, and2_activation])

    return orNeuron_activation


print("-------------- Output -----------------")
# print("🌟", network_ouput([0, 0]))
print("🌟", network_ouput([0, 1]))
# print("🌟", network_ouput([1, 0]))
# print("🌟", network_ouput([1, 1]))


# print("Start Weight", neuron.weights)
# for epoch in range(1000):
#     print("epoch:", epoch)
#     for i, x in enumerate(x_data):
#         label = y_labels[i]
#         neuron.learn(x, label, learning_rate=0.01)
