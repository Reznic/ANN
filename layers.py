"""Ann layers module."""
import numpy as np

from utils import SIGMOID
from errors import LayerInputSizeError


class Layer(object):
    """Neural Network layer.

    Attributes:
        size (nubmer): neurons in the layer.
        weights (numpy.ndarray): synapse weight factors,
            for each neuron in the layer.
        activation (func): the neurons activation function.
    """
    WEIGHT_MAX = 0.1
    WEIGHT_MIN = 0.001
    INPUT_SIZE = NotImplemented

    def __init__(self, size, next_layer=None, weights=NotImplemented,
                 activation=SIGMOID.func):
        self.size = size
        self.next_layer = next_layer
        self.weights = weights
        self.activation = activation

    def set_next(self, next_layer):
        self.next_layer = next_layer

    def feed_forward(self, inputs):
        """Stimulate inputs through the layer.
        
        Args:
            inputs (numpy.ndarray): input vector to feed_forward the
                layer neurons with.

        Returns:
            numpy.ndarray. vector of output value, from every neuron 
                in the layer.

        Raises:
            LayerInputSizeError: if inputs are of invalid size.
        """
        try:
            product = self.weights.dot(self._add_bias(inputs))
        except ValueError as error:
            raise LayerInputSizeError(error.message)

        return self.activation(product)

    def _gradient_descent(self, derivatives):
        """ """
        self.weights -= self.alpha * derivatives

    def back_propagation(self, inputs, expected_y):
        """ """
        y = self.feed_forward(inputs)

        # Todo: what if self.next_layer == None?
        accum_deriv = self.next_layer.back_propagation(y, expected_y)
        accum_deriv *= self.activation.deriv(y)

        # derivatives calculation
        derivs = np.row_stack(accum_deriv) * self._add_bias(inputs)

        # weights update
        self._gradient_descent(derivs)

        return accum_deriv

    def init_weights(self, input_size=None, min_w=WEIGHT_MIN,
                     max_w=WEIGHT_MAX):
        """Initialize layer weight factors.

        Args:
            input_size (number): layer input dimension.
            min_w (number): minimum limit for weights initialization.
            max_w (number): maximum limit for weights initialization.
        """
        if input_size is None:
            input_size = self.INPUT_SIZE

        # Add a bias weight to each neuron
        weights_per_neuron = input_size + 1

        self.weights = np.random.rand(self.size, weights_per_neuron) \
            * (max_w - min_w) + min_w

    def get_biases(self):
        """
        return: list. The biases of this layer.
        """
        return self.weights[:,[0]].flatten()

    @staticmethod
    def _add_bias(input):
        return np.append(np.array([-1]), input)


class InputLayer(Layer):
    """ANN input layer."""
    INPUT_SIZE = 1

    def feed_forward(self, inputs):
        """Stimulate inputs through the layer.

        Args:
            inputs (numpy.ndarray): input vector to feed_forward the
                layer neurons with.

        Returns:
            numpy.ndarray. vector of output value, from every neuron
                in the layer.

        Raises:
            LayerInputSizeError: if inputs are of invalid size.
        """
        try:
            # multiply input vector with weights of input layer (second column),
            # and subtract the biases vector (first columnt).
            product = inputs * self.weights[:,[1]].flatten() - self.get_biases()

        except ValueError as error:
            raise LayerInputSizeError(error.message)

        return self.activation(product)

    def back_propagation(self, inputs, expected_y):
        """ """
        y = self.feed_forward(inputs)

        accum_deriv = (y - expected_y) * self.activation.deriv(y)

        # derivatives calculation
        derivs = np.row_stack(accum_deriv) * self._add_bias(inputs)

        # weights update
        self._gradient_descent(derivs)

        # Return W * (y-Y)*y*(1-y)
        return self.weights * accum_deriv[:, np.newaxis]

