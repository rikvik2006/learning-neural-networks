from typing import Callable, List, Tuple, Union

from neurons import ActivationFunctions, McCullochPitsNeuron


class NeuralNetwork:
    def output(
        self,
        inputs: List[Union[int, float]],
        activationFunction: Callable[[float], Union[int, float]],
    ) -> Union[int, float]:
        activation1, activation2 = self.frist_layer()

    def frist_layer(
        self, inputMatrix: List[List[Union[int, float]]]
    ) -> Tuple[Union[int, float], ...]:
        self.neuron1 = McCullochPitsNeuron()
        self.neuron2 = McCullochPitsNeuron()

        activation1 = self.neuron1.output(
            inputMatrix[0], ActivationFunctions.sigmoidFunction
        )
        activation2 = self.neuron2.output(
            inputMatrix[1], ActivationFunctions.sigmoidFunction
        )

        return (activation1, activation2)
