import random
from typing import List, Callable

from unusedcode.neuralNetwork import NeuralNetwork
from rosenblattPerception import RosenblattPerception
from rosenblattPerception import ActivationFunctions


class XORNetwork(NeuralNetwork):
    def output(
        self,
        activationFunction: Callable[[float], int | float],
        *inputs: List[int | float]
    ) -> int | float:
        # return super().output(activationFunction, *inputs)
        activation1, activation2 = self.__layer(
            2, ActivationFunctions.heavisideSteupFunction, *inputs
        )
