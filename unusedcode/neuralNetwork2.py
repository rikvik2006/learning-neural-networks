import random
from typing import Union, Callable, List
from rosenblattPerception import RosenblattPerception
from rosenblattPerception import ActivationFunctions


class NeuralNetwork:
    def __init__(
        self, n_input: int, n_hidden: int, n_output: int, debug: bool = False
    ) -> None:
        self.debug = debug
        self.hidden_layer = self.__create_layer(n_hidden, n_input)
        self.output_layer = self.__create_layer(n_output, n_hidden)

    def __create_layer(
        self,
        n_neurons: int,
        n_inputs: int,
        activation_function: Callable[
            [float], Union[int, float]
        ] = ActivationFunctions.heavisideSteupFunction,
    ) -> List[RosenblattPerception]:
        layer = []
        for _ in range(n_neurons):
            weights = [random.random() for _ in range(n_inputs + 1)]  # +1 for bias
            neuron = RosenblattPerception(weights, activation_function)
            layer.append(neuron)
        return layer

    def output(self, inputs: List[Union[int, float]]) -> List[Union[int, float]]:
        hidden_outputs = [neuron.output(inputs) for neuron in self.hidden_layer]
        return [neuron.output(hidden_outputs) for neuron in self.output_layer]

    def train(
        self,
        inputs: List[List[Union[int, float]]],
        labels: List[List[Union[int, float]]],
        learning_rate: float,
        epochs: int,
    ):
        for epoch in range(epochs):
            for input_data, label in zip(inputs, labels):
                self.__train_single(input_data, label, learning_rate)

    def __train_single(
        self,
        input_data: List[Union[int, float]],
        label: List[Union[int, float]],
        learning_rate: float,
    ):
        hidden_outputs = [neuron.output(input_data) for neuron in self.hidden_layer]
        output = [neuron.output(hidden_outputs) for neuron in self.output_layer]

        # Calculate output layer errors
        output_errors = [label[i] - output[i] for i in range(len(output))]

        # Update output layer weights
        for i, neuron in enumerate(self.output_layer):
            neuron.learn(hidden_outputs, output_errors[i], learning_rate)

        # Calculate hidden layer errors
        hidden_errors = [0] * len(self.hidden_layer)
        for i, neuron in enumerate(self.output_layer):
            for j in range(len(neuron.weights) - 1):  # Exclude bias
                hidden_errors[j] += neuron.weights[j + 1] * output_errors[i]

        # Update hidden layer weights
        for i, neuron in enumerate(self.hidden_layer):
            neuron.learn(input_data, hidden_errors[i], learning_rate)
