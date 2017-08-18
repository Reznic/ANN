"""Artifitial neural network module."""

class Ann(object):
    """Artificial neural network.

    Attributes:
        layers (list): the neuron layers in the network.
    """
    def __init__(self, *layers):
        self.layers = layers
        self.init_layers_weights()

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

    def get_output(self, inputs):
        """Calculate output of the ANN, for a given input vector.
        
        Args:
            inputs (numpy.ndarray): input vector for the ANN.
            
        Returns:
            numpy.ndarray. output vector from the ANN.
        """
        for layer in self.layers:
            inputs = layer.stimulate(inputs)

        return inputs

