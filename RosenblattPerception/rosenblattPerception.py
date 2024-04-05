from typing import Callable, List, Union


class RosenblattPerception:
    # The frist weight is the Bias
    weights = [0, 0, 0]

    def output(
        self,
        inputs: List[Union[int, float]],
        activation_function: Callable[[float], Union[int, float]],
    ):
        # Initialize the weighted
        weighted_sum = self.weights[0]

        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i + 1]
        return activation_function(weighted_sum)

    def learn(self, x, y_label):
        y_obitained = self.output(x)
        error = y_label - y_obitained
        print("input:", x, "y_label", y_label, "error", error)
        print(f"💡 input: {x}, y_ottenuto: {y_obitained}")
        learningRate = 0.01

        # aggiorno peso del bias
        self.weights[0] += learningRate * error

        # aggiorno i pesi degli input del perceptron
        for i in range(len(self.weights) - 1):
            self.weights[i + 1] += learningRate * error * x[i]
