import numpy as np
from torch import flatten
np.seterr(all='raise')


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
        ephsilon = 1e-15 # to avoid returning 1
        X = X.clip(-500, 500) # to prevent overflows

        return np.divide(1, 1 + np.exp(-X) + ephsilon)

    def __forward(self, input: np.matrix, thetas: list[np.matrix]) -> tuple[np.matrix, list[np.matrix]]:
        result: np.matrix = input
        layer_values: list[np.matrix] = [input]

        for i in range(len(self.layers) - 1):
            # z(i)
            result = np.dot(result, thetas[i].T)
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
        return self.__forward(input, self.thetas)
        # return to_one_hot(np.asarray(predictions.argmax(axis=1)).ravel(), num_chars)

    def cost(self, X: np.matrix, Y: np.matrix, lambd: float, thetas: list[np.matrix]) -> float:
        h, _ = self.__forward(X, thetas)

        # compute cost
        first_term = np.multiply(Y, np.log(h))
        second_term = np.multiply((1 - Y), np.log(1 - h))
        J = np.sum(first_term + second_term) / (-X.shape[0])

        # add regularization
        regularized = 0
        for thetas in thetas:
            regularized += np.sum(np.power(thetas, 2)) * (lambd / (2 * X.shape[0]))

        J += regularized

        return J
    
    # backpropagation
    def gradients(self, X: np.matrix, Y: np.matrix, lambd: float) -> tuple[list[np.matrix], list[np.matrix]]:
        m = X.shape[0]

        h, a_values = self.__forward(X, self.thetas)
        bias_a = [np.ones((m, 1)) for _ in a_values]

        grads = [np.zeros(theta.shape) for theta in self.thetas]
        biases = [np.zeros(bias.shape) for bias in self.biases]

        # last layer gradients
        last_delta = h - Y
        grads[-1] = np.dot(last_delta.T, a_values[-1]) / m
        grads[-1] += (lambd * self.thetas[-1]) / m  # regulatization

        biases[-1] = np.dot(last_delta.T, bias_a[-1]) / m

        # hidden layer gradients
        for i in reversed(range(1, len(a_values))):
            delta = np.multiply(np.dot(last_delta, self.thetas[i]), self.__ReLU_grad(a_values[i]))
            last_delta = delta

            grads[i - 1] = np.dot(delta.T, a_values[i - 1]) / m
            grads[i - 1] += (lambd * self.thetas[i - 1]) / m   # regulatization

            biases[i - 1] = np.dot(delta.T, bias_a[i - 1]) / m

        return grads, biases

    def gradient_check(self, X: np.matrix, Y: np.matrix, lambd: float, grads: list[np.matrix]) -> bool:
        ephsilon = 1e-5
        
        new_theta1 = [np.copy(theta) for theta in self.thetas]
        new_theta2 = [np.copy(theta) for theta in self.thetas]
        preds = [np.zeros(grad.shape) for grad in grads]
        for l in range(len(self.thetas)):
            for i in range(self.thetas[l].shape[0]):
                for j in range(self.thetas[l].shape[1]):
                    new_theta1[l][i, j] = new_theta1[l][i, j] + ephsilon
                    new_theta2[l][i, j] = new_theta2[l][i, j] - ephsilon

                    grad_pred = self.cost(X, Y, lambd, new_theta1) - self.cost(X, Y, lambd, [theta for theta in new_theta2])
                    grad_pred = grad_pred / (2 * ephsilon)
                    preds[l][i, j] = grad_pred

                    new_theta1[l][i, j] = new_theta1[l][i, j] - ephsilon
                    new_theta2[l][i, j] = new_theta2[l][i, j] + ephsilon

        flattened_grads = np.concatenate([np.asarray(grad).ravel() for grad in grads])
        flattened_preds = np.concatenate([np.asarray(pred).ravel() for pred in preds])

        numerator = np.linalg.norm(flattened_grads - flattened_preds)
        denominator = np.linalg.norm(flattened_grads) + np.linalg.norm(flattened_preds)
        diff = numerator / denominator

        if diff > 1e-4:
            print(diff)
            print(grads) 
            print(preds)
            return False

        return True

    def train(self, X: np.matrix, Y: np.matrix, alpha: float, lambd: float, max_iters: int = 1000) -> np.ndarray:
        cost = np.zeros(max_iters)

        for i in range(max_iters):
            theta_grads, bias_grads = self.gradients(X, Y, lambd)

            assert self.gradient_check(X, Y, lambd, theta_grads)

            for j, _ in enumerate(self.thetas):
                self.thetas[j] = self.thetas[j] - np.multiply(alpha, theta_grads[j])
                self.biases[j] = self.biases[j] - np.multiply(alpha, bias_grads[j])

            cost[i] = self.cost(X, Y, lambd, self.thetas)

            if np.array_equal(self.predict(X)[0].round(), Y): break

        return cost[cost > 0]


if __name__ == "__main__":
    # X -> Y = logical OR operation
    X = np.asmatrix([[0, 0], [1, 0], [0, 1], [1, 1]])
    Y = np.asmatrix([[0], [1], [1], [1]])

    nn = NeuralNetworkModel(2, 1, [50, 5])
    nn.train(X, Y, 0.01, 0, 10)

    res, val = nn.predict(X)
    print(res)