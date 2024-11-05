import unittest
import numpy as np
import main


# Testy jednostkowe
class TestNeuralNetwork(unittest.TestCase):
    def test_sigmoid(self):
        # Test dla funkcji sigmoid
        self.assertAlmostEqual(main.sigmoid(0), 0.5)
        self.assertAlmostEqual(main.sigmoid(1), 0.7310585786300049, places=7)

    def test_sigmoid_derivative(self):
        # Test dla pochodnej sigmoida
        self.assertAlmostEqual(main.sigmoid_derivative(0.5), 0.25)
        self.assertAlmostEqual(main.sigmoid_derivative(0.7310585786300049), 0.19661193324148185, places=7)

    def test_forward_propagation(self):
        # Test propagacji w przód
        nn = main.NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        output = nn.forward(X)
        # Sprawdzamy, czy wynik ma odpowiedni kształt
        self.assertEqual(output.shape, (4, 1))

    def test_training_xor(self):
        # Testowanie, czy sieć nauczy się problemu XOR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        nn = main.NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
        nn.train(X, y, epochs=10000)
        
        # Sprawdzamy, czy wynik jest zgodny z oczekiwanym wzorcem XOR
        output = nn.forward(X)
        expected = np.array([[0], [1], [1], [0]])
        for i in range(len(output)):
            # Zakładamy, że sieć nauczyła się XOR jeśli wyniki są bliskie oczekiwanym
            self.assertAlmostEqual(output[i][0], expected[i][0], delta=0.1)

if __name__ == "__main__":
    unittest.main()    