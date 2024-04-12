from abc import ABC, abstractmethod
from typing import Callable, List, Union
from neurons import McCullochPitsNeuron


class NeuralNetwork(ABC):

    def __init__(self, debug: bool) -> None:
        self.debug = debug

    @abstractmethod
    def output(
        self,
        activationFunction: Callable[[float], Union[int, float]],
        *inputs: List[Union[int, float]],
    ) -> Union[int, float]:
        pass

    def __layer(
        self,
        n_neurons: int,
        activationFunction: Callable[[float], Union[int, float]],
        *inputMatrix: List[Union[int, float]],
    ) -> List[Union[int, float]]:
        """Questo metodo permette di definire un layer con n neuroni che utilizzano tutti una determinata activation function"""

        if self.debug:
            print(f"n_neurons: {n_neurons}, inputMatrix lenght: {len(inputMatrix)}\n")

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
