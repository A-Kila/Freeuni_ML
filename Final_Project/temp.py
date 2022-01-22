import numpy as np


class NeuralNetworkModel:
    def __init__(self, in_size: int, out_size: int, hidden_sizes: list[int]) -> None:
        self.layers: list[int] = [in_size] + hidden_sizes + [out_size]
        
        self.thetas = list[np.matrix]()
        self.biases = list[np.matrix]()

        for i in range(len(self.layers) - 1):
            prev_size: int = self.layers[i]
            next_size: int = self.layers[i + 1]

            self.thetas.append(np.matrix(np.random.rand(next_size, prev_size)))
            self.biases.append(np.matrix(np.random.rand(next_size, 1)))

    def __ReLU(self, X: np.matrix) -> np.matrix:
        func = np.vectorize(lambda x: max(0, x))
        return func(X)

    def __ReLU_grad(self, X: np.matrix) -> np.matrix:
        func = np.vectorize(lambda x: 1 if x > 0 else 0)
        return func(X)

    def sigmoid(self, X: np.matrix) -> np.matrix:
        return 1 / (1 + np.exp(-X))

    def __forward(self, input: np.matrix) -> tuple[np.matrix, list[np.matrix]]:
        result: np.matrix = input
        layer_values: list[np.matrix] = [input]

        for i in range(len(self.layers) - 1):
            # z(i)
            result = np.dot(result, self.thetas[i].T)
            result = np.add(result, self.biases[i].T)
            # a(i)
            if i != len(self.layers) - 2: 
                result = self.__ReLU(result) 
                layer_values.append(result)

            else: result = self.sigmoid(result)

        return result, layer_values

    # forward propagation
    def predict(self, input: np.matrix) -> np.matrix:
        return self.__forward(input)[0]

    def cost(self, X: np.matrix, Y: np.matrix, lambd: float) -> None:
        h, _ = self.__forward(X)

        first_term = np.multiply(Y, np.log(h))
        second_term = np.multiply((1 - Y), np.log(1 - h))
        J = np.sum(first_term + second_term) / (-X.shape[0])

        return J
    
    # backpropagation
    def gradients(self, X: np.matrix, Y: np.matrix, lambd: float) -> tuple[list[np.matrix], list[np.matrix]]:
        h, a_values = self.__forward(X)
        bias_a = [np.ones((X.shape[0], 1)) for _ in a_values]

        grads: list[np.matrix] = [np.zeros(theta.shape) for theta in self.thetas]
        biases: list[np.matrix] = [np.zeros(bias.shape) for bias in self.biases]

        last_delta: np.matrix = np.subtract(h, Y)
        grads[-1] = (grads[-1] + np.dot(last_delta.T, a_values[-1])) / X.shape[0]
        biases[-1] = (biases[-1] + np.dot(last_delta.T, bias_a[-1])) / X.shape[0]

        for i in range(len(a_values) - 1, 0, -1):
            delta = np.multiply(np.dot(last_delta, self.thetas[i]), self.__ReLU_grad(a_values[i]))
            grads[i - 1] = (grads[i - 1] + np.dot(delta.T, a_values[i - 1]))
            biases[i - 1] = (biases[i - 1] + np.dot(delta.T, bias_a[i - 1]))

            last_delta = delta

        return grads, biases

    def train(self) -> None:
        pass # TODO


if __name__ == "__main__":
    nn = NeuralNetworkModel(2, 2, [3, 2])
    print(nn.predict(np.matrix([3, 3])))
    print(nn.cost(np.matrix([[3, 3], [2, 2], [4, 4]]), np.matrix([[1, 0], [0, 1], [0, 0]]), 0))
    print(nn.gradients(np.matrix([[3, 3], [2, 2], [4, 4]]), np.matrix([[1, 0], [0, 1], [0, 0]]), 0))
