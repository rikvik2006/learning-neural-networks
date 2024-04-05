import math
from typing import Callable, List, Union


class ActivationFunctions:

    @staticmethod
    def sigmoidFunction(weightedSum: float) -> float:
        return 1 / (1 + math.exp(-weightedSum))

    @staticmethod
    def heavisideSteupFunction(weightedSum: float) -> int:
        if weightedSum < 0:
            return 0
        else:
            return 1


class McCullochPitsNeuron:
    weights = [0.5, 0.5]

    def output(
        self,
        inputs: List[Union[int, float]],
        activationFunction: Callable[[float], Union[int, float]],
    ):
        weightedSum = 0
        for i in range(len(inputs)):
            weightedSum += inputs[i] * self.weights[i]
        return activationFunction(weightedSum)


if __name__ == "__main__":
    neuron = McCullochPitsNeuron()
    print(neuron.output([0, 0], ActivationFunctions.sigmoidFunction))
    print(neuron.output([0, 1], ActivationFunctions.sigmoidFunction))
    print(neuron.output([1, 1], ActivationFunctions.sigmoidFunction))
