"""Ann unitest module."""
import sys
import numpy as np
from unittest import (TestCase, TestSuite, TextTestRunner, 
                      defaultTestLoader)

sys.path.append('../ann')
from ann import Ann
from layers import Layer, InputLayer


class BaseAnnTest(TestCase):
    def assert_array_equal(self, arr1, arr2, fail_msg=None, precision=1e-6):
        """Assert two arrays are equal.

        Args:
            arr1 (numpy.ndarray): first array.
            arr2 (numpy.ndarray): second array.
            precision (float): scalars comparison precision treshold.
            fail_msg (str): message to present, if assertion fails.
        """
        if fail_msg is None:
            fail_msg = "%s differes from %s" % (arr1, arr2)

        deltas = abs(arr1 - arr2)
        thresholds = deltas - precision
        self.assertFalse(any(thresholds > 0), fail_msg)


class BiasesTest(BaseAnnTest):
    """Test the ann's biases mechanism."""
    HIDDEN_LAYER_SIZE = 3
    INPUT = np.array([1,2,3,4])
    EXPECTED_OUTPUT = np.array([0, 0])
    WEIGHTS = (np.array([[100, 0.1], [100, 0.2], [100, 0.3], [100, 0.4]]),
               np.array([[100, 0.1, 0.1, 0.2, 0.01], [100, 0.5, 0.4, 0.8, 0.3],
                         [100, 0.12, 0.43, 0.5, 0.002]]),
               np.array([[100, 0.24,0.12,0.6], [100, 0.1,0.004,0.21]]))

    def setUp(self):
        self.ann = Ann(InputLayer(self.INPUT.size, self.WEIGHTS[0]),
                       Layer(self.HIDDEN_LAYER_SIZE,
                             self.WEIGHTS[1]),
                       Layer(self.EXPECTED_OUTPUT.size,
                             self.WEIGHTS[2]))

    def test_large_biases(self):
        output = self.ann.feed_forward(self.INPUT)
        self.assert_array_equal(output, self.EXPECTED_OUTPUT,
                                "Wrong forward output: %s instead of"
                                " %s" % (output, self.EXPECTED_OUTPUT))

    def test_small_and_large_biases(self):
        ann = Ann(InputLayer(self.INPUT.size),
                       Layer(self.HIDDEN_LAYER_SIZE),
                       Layer(self.EXPECTED_OUTPUT.size))

        # Set output layer's 2 neurons biases:
        ann.layers[-1].weights[0][0] = 100
        ann.layers[-1].weights[1][0] = -100

        expected_output = np.array([0, 1])
        output = ann.feed_forward(self.INPUT)
        self.assert_array_equal(output, expected_output,
                                "Wrong forward output: %s instead of"
                                " %s" % (output, expected_output))

    def test_input_layer_biases(self):
        inputs_vector = np.array([10, 20])
        ann = Ann(InputLayer(inputs_vector.size))

        # Set output layer's 2 neurons biases:
        ann.layers[0].weights[0][0] = 100
        ann.layers[0].weights[1][0] = -100

        expected_output = np.array([0, 1])
        output = ann.feed_forward(inputs_vector)
        self.assert_array_equal(output, expected_output,
                                "Wrong forward output: %s instead of"
                                " %s" % (output, expected_output))

    def test_full_layer_biases(self):
        inputs_vector = np.array([10, 20])
        layer = Layer(inputs_vector.size)
        layer.init_weights(2)

        # Set output layer's 2 neurons biases:
        layer.weights[0][0] = 100
        layer.weights[1][0] = -100

        expected_output = np.array([0, 1])
        output = layer.feed_forward(inputs_vector)
        self.assert_array_equal(output, expected_output,
                                "Wrong forward output: %s instead of"
                                " %s" % (output, expected_output))


class ForwardTest(BaseAnnTest):
    """Test the ann's feed-forward flow"""
    HIDDEN_LAYER_SIZE = 3
    INPUT = np.array([1,2,3,4])
    # Todo: add biases, and calculate output
    WEIGHTS = (np.array([[0, 0.1], [0, 0.2], [0, 0.3], [0, 0.4]]),
               np.array([[0, 0.1, 0.1, 0.2, 0.01], [0, 0.5, 0.4, 0.8, 0.3],
                         [0, 0.12, 0.43, 0.5, 0.002]]),
               np.array([[0, 0.24,0.12,0.6], [0, 0.1,0.004,0.21]]))

    EXPECTED_OUTPUT = np.array([0.65210587, 0.5495772])

    def setUp(self):
        self.ann = Ann(InputLayer(self.INPUT.size, self.WEIGHTS[0]), 
                       Layer(self.HIDDEN_LAYER_SIZE, 
                             self.WEIGHTS[1]),
                       Layer(self.EXPECTED_OUTPUT.size, 
                             self.WEIGHTS[2]))

    def test_full_forward(self):
        output = self.ann.feed_forward(self.INPUT)
        self.assert_array_equal(output, self.EXPECTED_OUTPUT,
                                "Wrong forward output: %s instead of"
                                " %s" % (output,
                                         self.EXPECTED_OUTPUT))

    def test_hidden_layer_forward(self):
        layer_input = np.array([1,2,0.5,0])
        expected_output = np.array([0.598687660112452, 
                                    0.84553473491646525, 
                                    0.77381857426945377])

        layer_output = self.ann.layers[1].feed_forward(layer_input)
        self.assert_array_equal(layer_output, expected_output,
                                "Wrong hidden layer output. %s "
                                "instead %s" % (layer_output, 
                                                expected_output))


class RandWeightsTest(BaseAnnTest):
    """Test ANN random weights initialization."""
    LAYER_SIZES = [3,7,5]

    def setUp(self):
        self.ann = Ann(InputLayer(self.LAYER_SIZES[0]), 
                       Layer(self.LAYER_SIZES[1]),
                       Layer(self.LAYER_SIZES[2]))

    def test_rand_weights_sizes(self):
        initialized_sizes = [layer.weights.shape
                             for layer in self.ann.layers]

        expected_sizes = zip(self.LAYER_SIZES, 
                             [s+1 for s in ([1] + self.LAYER_SIZES)])

        self.assertEqual(initialized_sizes, expected_sizes,
                         "ANN's layers initialized with wrong sizes: %s "
                         "instead of %s" % (initialized_sizes, expected_sizes))

    def test_rand_weights_range(self):
        for layer in self.ann.layers:
            for weight in layer.weights.flatten():
                self.assertTrue((Layer.WEIGHT_MIN <= weight < Layer.WEIGHT_MAX),
                                "A random weight %s initialized "
                                "out of the range: [%s, %s)"
                                % (weight, Layer.WEIGHT_MAX,
                                   Layer.WEIGHT_MIN))


class AnnTestSuite(TestSuite):
    TESTS = (ForwardTest,
             RandWeightsTest,
             BiasesTest)

    def __init__(self):
        super(AnnTestSuite, self).__init__()
        for test in self.TESTS:
            self.addTest(defaultTestLoader.\
                         loadTestsFromTestCase(test))


if __name__ == "__main__":
    TextTestRunner().run(AnnTestSuite())

