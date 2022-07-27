import numpy as np
import time

class Perceptron:
    
    def __init__(self) -> None:
        np.random.seed(100)
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x) -> float:
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x) -> float:
        return x * (1 - x)

    def train(self, training_input, training_output, training_iteration) -> None:
        
        for i in range(training_iteration):
            output = self.predict(training_input)
            print("outputs")
            print(output)
            error = training_output - output
            adjustments = np.dot(training_input.T, error * self.sigmoid_deriv(output))
            print("adjustment")
            print(adjustments)
            self.weights += adjustments
            #time.sleep(1)

        print("Done training")

    def predict(self, input):
        input = input.astype(float)
        output = self.sigmoid(np.dot(input, self.weights))

        return output

if __name__ == "__main__":
    
    nn = Perceptron()

    print("weights before training")
    print(nn.weights)

    training_inputs = np.array([[0,0,1],[1,1,1], [1,0,1], [0,1,1]])
    training_outputs = np.array([[0,1,1,0]]).T

    nn.train(training_inputs, training_outputs, 100)

    print("weights after training")
    print(nn.weights)

    test_input = np.array([1, 1, 0])

    print("prediction is ")
    print(nn.predict(test_input))