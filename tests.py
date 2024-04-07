from network import *
import unittest

class TestActivationFunctions(unittest.TestCase):
    def test_ReLU(self):
        relu = ReLU()
        
        relu_in = np.array([-5, 5, -4, 4, -3, 3, -2, 2, -1, 1])
        relu_out = relu.forward(relu_in)
        desired_out = np.array([0, 5, 0, 4, 0, 3, 0, 2, 0, 1])
        relu.backward(np.ones_like(relu_out))
        relu_grad = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.assertTrue(np.all(relu_out == desired_out))
        self.assertTrue(np.all(relu.grad == relu_grad))

class TestLayer(unittest.TestCase):
    def test_layer(self):
        layer = Layer(3, 3)
        layer_in = np.expand_dims(np.array([1, 2, 3]), axis=1)
        layer.weights = np.array([
            [0, 1, -1],
            [1, -1, 1],
            [2, -1, -2],
            [-1, 1, 0]
        ])
        layer_out = layer.forward(layer_in)
        desired_out = np.array([7, -3, -5])
        self.assertTrue(np.all(layer_out == desired_out))

        next_grad = np.array([1, 2, 3])
        grad_out = layer.backward(next_grad)
        desired_grad_out = np.array([-1, 2, -6])
        desired_grad = np.array([
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9],
            [1, 2, 3]
        ])
        self.assertTrue(np.all(desired_grad == layer.grad))
        self.assertTrue(np.all(grad_out == desired_grad_out))

class TestSequential(unittest.TestCase):
    def test_1(self):
        layer_1 = Layer(2, 3)
        layer_1.weights = np.array([
            [1, 7, 4],
            [3, 5, 6],
            [2, 8, 3]
        ])

        layer_2 = Layer(3, 1)
        layer_2.weights = np.array([
            [2],
            [1],
            [-1],
            [1]
        ])
        model = Sequential([
            layer_1,
            ReLU(),
            layer_2
        ],
        criterion=None)

        model_in = np.expand_dims(np.array([-1, 2]), axis=1)
        model_out = model.forward(model_in)
        desired_out = np.array([15])
        self.assertTrue(np.all(model_out == desired_out))


if __name__ == "__main__":
    unittest.main()
