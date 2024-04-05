from typing import Callable, List, Tuple, Union

from neurons import ActivationFunctions, McCullochPitsNeuron


class NeuralNetwork:

    def __init__(self, debug: bool) -> None:
        self.debug = debug

    def output(
        self,
        activationFunction: Callable[[float], Union[int, float]],
        *inputs: List[Union[int, float]],
    ) -> Union[int, float]:
        # for data in inputs:
        #     if len(data) > 2:
        #         raise ValueError(
        #             "You can only give 2 input data to every single neuron"
        #         )

        activation1, activation2 = self.__layer(2, activationFunction, *inputs)
        outputActivation = self.__layer(
            1,
            activationFunction,
            [activation1, activation2],
        )

        if self.debug:
            print(
                f"1° Layer activations: Neuron 1: {activation1}, Neuron 2: {activation2}"
            )
            print(f"2° Layer activations: Neuron 1: {outputActivation}\n")

        return outputActivation[0]

    def __layer(
        self,
        n_neurons: int,
        activationFunction: Callable[[float], Union[int, float]],
        *inputMatrix: List[Union[int, float]],
    ) -> List[Union[int, float]]:
        """Questo metodo permette di definire un layer con n neuroni che utilizzano tutti una determinata activation function"""

        if self.debug:
            print(
                f"n_neurons: {n_neurons}, inputMatrix lenght: {len(inputMatrix)}\n"
            )

        if n_neurons != len(inputMatrix):
            raise ValueError(
                "n_neurons and inputMatrix lenght must be the same valu es: n_neurons = 5, len(inputMatrix) -> 5"
            )

        self.neurons: List[McCullochPitsNeuron] = []
        self.activations: List[Union[int, float]] = []

        for i in range(n_neurons):
            neuron = McCullochPitsNeuron()
            self.neurons.append(neuron)
            activation = neuron.output(inputMatrix[i], activationFunction)
            self.activations.append(activation)

        return self.activations


neural_network = NeuralNetwork(debug=True)


print("🎯 Funzione di attivazione sigmoide")

print("----Dati binari----")
output = neural_network.output(
    ActivationFunctions.sigmoidFunction, [0, 0], [1, 0]
)
print(f"🌟 Output neurone: {output}\n--------\n")
output = neural_network.output(
    ActivationFunctions.sigmoidFunction, [0, 0], [1, 0]
)
print(f"🌟 Output neurone: {output}\n--------\n")

print("\n----Dati in R----")
output = neural_network.output(
    ActivationFunctions.sigmoidFunction, [3, -2, 5.4], [0.1, -5, 3.2]
)
print(f"🌟 Output neurone: {output}\n--------\n")
output = neural_network.output(
    ActivationFunctions.sigmoidFunction, [3, -2], [0.1, -5]
)
print(f"🌟 Output neurone: {output}\n--------\n")

print("\n🎯 Funzione di attivazione a gradino di Heaviside")

print("----Dati binari----")
output = neural_network.output(
    ActivationFunctions.heavisideSteupFunction,
    [0, 0],
    [1, 0],
)
print(f"🌟 Output neurone: {output}\n--------\n")
output = neural_network.output(
    ActivationFunctions.heavisideSteupFunction,
    [0, 0],
    [1, 0],
)
print(f"🌟 Output neurone: {output}\n--------\n")

print("\n----Dati in R----")
output = neural_network.output(
    ActivationFunctions.heavisideSteupFunction,
    [3, -2],
    [0.1, -5],
)
print(f"🌟 Output neurone: {output}\n--------\n")
output = neural_network.output(
    ActivationFunctions.heavisideSteupFunction,
    [3, -2],
    [0.1, -5],
)
print(f"🌟 Output neurone: {output}\n--------\n")
