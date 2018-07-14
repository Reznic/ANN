"""Artifitial neural network module."""

class Ann(object):
    """Artificial neural network.

    Attributes:
        layers (list): the neuron layers in the network.
    """

    def __init__(self, *layers):
        self.layers = layers
        self._connect_layers()
        self.init_layers_weights()

    def _connect_layers(self):
        reduce(self.layers,
               lambda l1, l2: l1.set_next(l2); return l2)
        self.layers[-1].set_next(None)

    def init_layers_weights(self):
        """Initialize weight factors in all the net layers."""
        # Init input layer weights
        if self.layers[0].weights is NotImplemented:
            self.layers[0].init_weights()

        # Init every layer, with respect to the previous layer size.
        adjacent_layers = zip(self.layers, self.layers[1:])
        for prev_layer, layer in adjacent_layers:
            if layer.weights is NotImplemented:
                layer.init_weights(prev_layer.size)

    def feed_forward(self, inputs):
        """Calculate output of the ANN, for a given input vector.
        
        Args:
            inputs (numpy.ndarray): input vector for the ANN.
            
        Returns:
            numpy.ndarray. output vector from the ANN.
        """
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)

        return inputs


    def back_propagate(self, inputs, expected_output):
        """ """
        ys = [inputs]
        gradients = []
        accumulated_gradient = None
        
        # Forward pass
        for layer in self.layers:
            ys.append(layer.feed_forward(ys[-1]))

        # Gradient calculation
        accumulated_gradient = (ys[-1] - expected_output)
        for layer, y in zip(reversed(self.layers), reversed(ys)):
            accumulated_gradient *= layer.activation.deriv(y)

        # Weights update
        for grads, layer in zip(gradients, reversed(self.layers)):
            # grad_descent(layer.weights, grads)
            layer.weights - self.alpha * grads

