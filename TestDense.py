import unittest
import numpy as np
from DenseLayer import Dense  # Replace 'your_module' with the actual module name

class TestDense(unittest.TestCase):
    def setUp(self):
        # Initialize test inputs and expected outputs
        self.inp = np.array([[1], [2]])  # Input with shape (2, 1)
        self.neurons = 3  # Number of neurons
        self.layer = Dense(self.inp, self.neurons)
        self.layer.inp = self.inp  # Directly set input for testing

    def test_forward(self):
        # Test the forward pass
        output = self.layer.forward()
        self.assertEqual(output.shape, (self.neurons, 1))
        self.assertTrue(np.all(output >= 0))  # Check ReLU activation

    def test_backward(self):
        # Test the backward pass
        gradient = np.array([[0.5], [0.2], [0.1]])  # Example gradient
        learning_rate = 0.01
        output = self.layer.forward()
        dI = self.layer.backward(gradient, learning_rate)

        # Check the shape of the returned gradient
        self.assertEqual(dI.shape, (self.inp.shape[0], 1))

        # Check the weights and biases have been updated
        self.assertTrue(np.any(self.layer.weights != np.random.randn(self.neurons, self.inp.shape[0])))
        self.assertTrue(np.any(self.layer.bias != np.random.randn(self.neurons, 1)))

if __name__ == '__main__':
    unittest.main()
