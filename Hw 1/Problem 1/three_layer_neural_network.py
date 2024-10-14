__author__ = 'tan_nguyen'

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt


def generate_data():
    """
    generate data
    :return: arr_x: input data, y: given labels
    """
    np.random.seed(0)
    arr_x, y = datasets.make_moons(200, noise=0.20)
    return arr_x, y


def plot_decision_boundary(pred_func, arr_x, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param arr_x: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = arr_x[:, 0].min() - .5, arr_x[:, 0].max() + .5
    y_min, y_max = arr_x[:, 1].min() - .5, arr_x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(arr_x[:, 0], arr_x[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show() #


########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """

    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, act_fun_type='tanh', reg_lambda=0.01, seed=0):
        """
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param act_fun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        """

        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.act_fun_type = act_fun_type
        self.reg_lambda = reg_lambda

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_hidden_dim, self.nn_input_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_output_dim, self.nn_hidden_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

        # Last dimension depend on input size
        self.probs = np.random.randn(self.nn_output_dim, 0) / np.sqrt(self.nn_output_dim)
        self.z2 = np.random.randn(self.nn_output_dim, 0) / np.sqrt(self.nn_output_dim)
        self.a1 = np.random.randn(self.nn_hidden_dim, 0) / np.sqrt(self.nn_hidden_dim)
        self.z1 = np.random.randn(self.nn_hidden_dim, 0) / np.sqrt(self.nn_hidden_dim)

    def act_fun(self, z, type):
        """
        act_fun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        """

        assert type in ['tanh', 'sigmoid', 'relu'], "type has to be 'tanh', 'sigmoid' or 'relu' "

        if type == 'tanh':
            return np.tanh(z)
        elif type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif type == 'relu':
            z_next = z.copy()
            z_next[z_next < 0] = 0
            return z_next

        raise NotImplementedError

    def diff_act_fun(self, z, type):
        """
        diff_actFun computes the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        """

        assert type in ['tanh', 'sigmoid', 'relu'], "type has to be 'tanh', 'sigmoid' or 'relu' "

        if type == 'tanh':
            return 1 / np.cosh(z) ** 2
        elif type == 'sigmoid':
            return np.exp(-z) / (1 + np.exp(-z)) ** 2
        elif type == 'relu':
            diff = np.ones(z.shape)
            diff[z < 0] = 0
            return diff

        raise NotImplementedError

    def feedforward(self, arr_x, act_fun):
        """
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param arr_x: input data
        :param act_fun: activation function
        :return:
        """
        n_samples = arr_x.shape[0]
        self.z1 = np.dot(self.W1, arr_x.T) + np.tile(self.b1.reshape((-1, 1)), (1, n_samples))
        self.a1 = act_fun(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + np.tile(self.b2.reshape((-1, 1)), (1, n_samples))
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    def calculate_loss(self, arr_x, y):
        """
        calculate_loss computes the loss for prediction
        :param arr_x: input data
        :param y: given labels
        :return: the loss for prediction
        """
        if type(y) == list:
            y = np.array(y)

        num_examples = arr_x.shape[0]
        self.feedforward(arr_x, lambda x: self.act_fun(x, type=self.act_fun_type))

        # Calculating the loss
        y_one_hot = np.concatenate(((1 - y).reshape((-1, 1)), y.reshape((-1, 1))), axis=1).T
        data_loss = - np.sum(y_one_hot * np.log(self.probs))

        # Add regularization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, arr_x):
        """
        predict infers the label of a given data point arr_x
        :param arr_x: input data
        :return: label inferred
        """
        self.feedforward(arr_x, lambda x: self.act_fun(x, type=self.act_fun_type))
        return np.argmax(self.probs, axis=0)

    def grad_softmax(self, z):
        """
        Compute the derivative of the final layer activation(s) softmax
        :param z: Last layer net values
        :return: The gradient of the softmax. Rows correspond to partial derivatives and columns correspond to the
         component of softmax
        """
        n_samples = z.shape[-1]
        exp_z = np.exp(z)
        exp_sum = np.sum(exp_z, axis=0, keepdims=True)[np.newaxis, :, :]

        exp_z_all_diag = np.apply_along_axis(np.diag, -1,
                                             exp_z.T).T  # First two dimensions make diagonal array and last dimension is the sample
        return (exp_sum * exp_z_all_diag - exp_z[np.newaxis, :, :] * exp_z[:, np.newaxis, :]) / exp_sum ** 2

    def calc_phi(self, arr_x):
        "Computes the coefficients for backprop"
        n_samples = arr_x.shape[0]
        self.phi_2 = np.tile(np.eye(self.nn_output_dim)[:, :, np.newaxis], (1, 1, n_samples))
        self.phi_1 = np.einsum('jpn, pq, qn -> jqn', self.phi_2, self.W2,
                               self.diff_act_fun(self.z1, type=self.act_fun_type))

    def backprop(self, arr_x, y):
        """
        backprop implements backpropagation to compute the gradients used to update the parameters in the backward step
        :param arr_x: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        """

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = arr_x.shape[0]
        # delta3 = self.probs
        # delta3[range(num_examples), y] -= 1

        # For each sample
        # print(self.grad_softmax(self.z2).shape, (y / self.probs).shape)
        mu_3 = -np.einsum('ijk, ik -> jk', self.grad_softmax(self.z2), (y / self.probs))

        dW2 = -np.einsum('jk, ik -> ij', self.a1, mu_3) / num_examples
        db2 = -np.sum(mu_3, axis=1) / num_examples

        self.calc_phi(arr_x)
        dW1 = np.einsum('im, ijn, jkn, rn -> krn', y / self.probs, self.grad_softmax(self.z2), self.phi_1, arr_x.T)
        dW1 = -np.sum(dW1, axis=-1) / num_examples
        db1 = np.einsum('im, ijn, jkn-> kn', y / self.probs, self.grad_softmax(self.z2), self.phi_1)
        db1 = -np.sum(db1, axis=-1) / num_examples

        return dW1, dW2, db1, db2

    def fit_model(self, arr_x, y, epsilon=0.01, num_passes=2000, print_loss=True): # 20_000
        """
        fit_model uses backpropagation to train the network
        :param arr_x: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        """
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(arr_x, lambda x: self.act_fun(x, type=self.act_fun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(arr_x, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(arr_x, y)))

    def visualize_decision_boundary(self, arr_x, y):
        """
        visualize_decision_boundary plots the decision boundary created by the trained network
        :param arr_x: input data
        :param y: given labels
        :return:
        """
        plot_decision_boundary(lambda x: self.predict(x), arr_x, y)


def main():
    # generate and visualize Make-Moons dataset
    arr_x, y = generate_data()
    plt.scatter(arr_x[:, 0], arr_x[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()

    model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3, nn_output_dim=2, act_fun_type='tanh')
    model.fit_model(arr_x, y)
    model.visualize_decision_boundary(arr_x, y)


if __name__ == "__main__":
    main()
