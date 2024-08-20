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


class RosenblattPerception:
    # Il primo peso è il Bias del neurone, gli altri pesi sono i pesi che saranno assegnati ad ogni rispettivo input
    def __init__(
        self,
        weights: List[Union[int, float]],
        activation_function: Callable[[float], Union[int, float]],
    ) -> None:
        self.weights = weights
        self.activation_function = activation_function

    def output(
        self,
        inputs: List[Union[int, float]],
    ):
        # Inizializziamo il valore della soma pesata prendendo il valore del BIAS
        weighted_sum = self.weights[0]

        # Calcoliamo la somma pesata saltando il Bias come peso dato che l'abbiamo già aggiunto
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i + 1]

        # Restituiamo il risultato della activation function
        return self.activation_function(weighted_sum)

    def learn(self, x, y_label, learning_rate: float):
        y_obitained = self.output(x)
        error = y_label - y_obitained
        print(
            f"💡 input: {x}, y_ottenuto: {y_obitained}, y_label: {y_label}, erorr: {error}"
        )
        learningRate = learning_rate

        # aggiorno peso del bias
        self.weights[0] += learningRate * error

        # aggiorno i pesi degli input del perceptron
        for i in range(len(self.weights) - 1):
            self.weights[i + 1] += learningRate * error * x[i]
