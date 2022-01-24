import numpy as np


class NeuralNetworkModel:
    def __init__(self, in_size: int, out_size: int, hidden_sizes: list[int]) -> None:
        self.layers: list[int] = [in_size] + hidden_sizes + [out_size]
        
        self.thetas = list[np.matrix]()
        self.biases = list[np.matrix]()

        for i in range(len(self.layers) - 1):
            prev_size: int = self.layers[i]
            next_size: int = self.layers[i + 1]

            self.thetas.append(np.asmatrix(np.random.rand(next_size, prev_size)))
            self.biases.append(np.asmatrix(np.random.rand(next_size, 1)))

    def __ReLU(self, X: np.matrix) -> np.matrix:
        return np.maximum(0, X)

    def __ReLU_grad(self, X: np.matrix) -> np.matrix:
        return np.where(X <= 0, 0, 1)

    def __sigmoid(self, X: np.matrix) -> np.matrix:
        ephsilon = 1e-5 # to avoid returning 1
        X = X.clip(-500, 500) # to prevent overflows

        return np.divide(1, 1 + np.exp(-X) + ephsilon)

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

            else: result = self.__sigmoid(result)

        return result, layer_values

    # forward propagation, but returns array with only 1 and 0s
    def predict(self, input: np.matrix) -> np.matrix:
        # predictions =
        return self.__forward(input)
        # return to_one_hot(np.asarray(predictions.argmax(axis=1)).ravel(), num_chars)

    def cost(self, X: np.matrix, Y: np.matrix, lambd: float) -> float:
        h, _ = self.__forward(X)

        # compute cost
        first_term = np.multiply(Y, np.log(h))
        second_term = np.multiply((1 - Y), np.log(1 - h))
        J = np.sum(first_term + second_term) / (-X.shape[0])

        # add regularization
        regularized = 0
        for thetas in self.thetas:
            regularized += np.sum(np.power(thetas, 2))

        J += np.multiply(np.divide(lambd, 2 * X.shape[0]), regularized)

        return J
    
    # backpropagation
    def gradients(self, X: np.matrix, Y: np.matrix, lambd: float) -> tuple[list[np.matrix], list[np.matrix]]:
        h, a_values = self.__forward(X)
        bias_a = [np.asmatrix(np.ones((X.shape[0], 1))) for _ in a_values]

        grads: list[np.matrix] = [np.zeros(theta.shape) for theta in self.thetas]
        biases: list[np.matrix] = [np.zeros(bias.shape) for bias in self.biases]

        last_delta: np.matrix = np.subtract(h, Y)
        grads[-1] = (grads[-1] + np.dot(last_delta.T, a_values[-1])) / X.shape[0]
        biases[-1] = (biases[-1] + np.dot(last_delta.T, bias_a[-1])) / X.shape[0]

        for i in range(len(a_values) - 1, 0, -1):
            delta = np.multiply(np.dot(last_delta, self.thetas[i]), self.__ReLU_grad(a_values[i]))

            grads[i - 1] = grads[i - 1] + np.dot(delta.T, a_values[i - 1]) / X.shape[0]
            grads[i - 1] = grads[i - 1] + np.multiply(lambd / X.shape[0], self.thetas[i - 1]) # regularization

            biases[i - 1] = (biases[i - 1] + np.dot(delta.T, bias_a[i - 1])) / X.shape[0]

            last_delta = delta

        return grads, biases

    def train(self, X: np.matrix, Y: np.matrix, alpha: float, lambd: float, max_iters: int = 1000) -> np.ndarray:
        cost = np.zeros(max_iters)

        for i in range(max_iters):
            # cost[i] = self.cost(X, Y, lambd)
            theta_grads, bias_grads = self.gradients(X, Y, lambd)

            for j, _ in enumerate(self.thetas):
                self.thetas[j] = self.thetas[j] - np.multiply(alpha, theta_grads[j])
                self.biases[j] = self.biases[j] - np.multiply(alpha, bias_grads[j])

        return cost


if __name__ == "__main__":
    # X -> Y = logical OR operation
    X = np.asmatrix([[0, 0], [1, 0], [0, 1], [1, 1]])
    Y = np.asmatrix([[0], [1], [1], [1]])

    nn = NeuralNetworkModel(2, 1, [150, 70])
    nn.train(X, Y, 0.1, 0.1, 10000)

    res, val = nn.predict(X)
    print(res)