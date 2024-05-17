import math

from neurons import ActivationFunctions, McCullochPitsNeuron

neuron = McCullochPitsNeuron(weight=[0.5, 0.5])
print("Funzione di attivazione sigmoide")
print("----Dati binari----")
print(neuron.output([0, 0], ActivationFunctions.sigmoidFunction))
print(neuron.output([0, 1], ActivationFunctions.sigmoidFunction))
print(neuron.output([1, 0], ActivationFunctions.sigmoidFunction))
print(neuron.output([1, 1], ActivationFunctions.sigmoidFunction))

print("\n----Dati in R----")
print(neuron.output([0.5, 1], ActivationFunctions.sigmoidFunction))
print(neuron.output([-3, 7], ActivationFunctions.sigmoidFunction))
print(neuron.output([-4.5, -0.02], ActivationFunctions.sigmoidFunction))
print(
    neuron.output([math.pi, math.sqrt(2)], ActivationFunctions.sigmoidFunction)
)

print("\nFunzione di attivazione a gradino di Heaviside")
print("----Dati binari----")
print(neuron.output([0, 0], ActivationFunctions.heavisideSteupFunction))
print(neuron.output([0, 1], ActivationFunctions.heavisideSteupFunction))
print(neuron.output([1, 0], ActivationFunctions.heavisideSteupFunction))
print(neuron.output([1, 1], ActivationFunctions.heavisideSteupFunction))

print("\n----Dati in R----")
print(neuron.output([0.5, 1], ActivationFunctions.heavisideSteupFunction))
print(neuron.output([-3, 7], ActivationFunctions.heavisideSteupFunction))
print(neuron.output([-4.5, -0.02], ActivationFunctions.heavisideSteupFunction))
print(
    neuron.output(
        [math.pi, math.sqrt(2)], ActivationFunctions.heavisideSteupFunction
    )
)
