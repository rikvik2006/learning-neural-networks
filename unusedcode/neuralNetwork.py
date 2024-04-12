import random
from typing import Union, Callable, List
from abc import ABC, abstractmethod
from rosenblattPerception import RosenblattPerception


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

        self.neurons: List[RosenblattPerception] = []
        self.activations: List[Union[int, float]] = []

        weights = []
        for i in range(0, len(inputMatrix[0])):
            weights.append(random.random())
        for i in range(n_neurons):
            neuron = RosenblattPerception(
                weights=weights, activation_function=activationFunction
            )
            self.neurons.append(neuron)
            activation = neuron.output(inputMatrix[i])
            self.activations.append(activation)

        return self.activations
