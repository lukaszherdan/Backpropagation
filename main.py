import numpy as np

# Funkcja aktywacji (tutaj Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji Sigmoid dla backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Implementacja sieci neuronowej
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicjalizacja wag z losowymi wartościami
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Wagi między warstwą wejściową a ukrytą
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        # Wagi między warstwą ukrytą a wyjściową
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    # Funkcja forward propagation
    def forward(self, X):
        # Przepływ sygnałów przez warstwę ukrytą
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        
        # Przepływ sygnałów przez warstwę wyjściową
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    # Funkcja backpropagation
    def backward(self, X, y, output):
        # Błąd na wyjściu
        error = y - output
        
        # Obliczenie gradientu błędu dla warstwy wyjściowej
        d_output = error * sigmoid_derivative(output)
        
        # Błąd dla warstwy ukrytej
        error_hidden_layer = d_output.dot(self.weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_output)
        
        # Aktualizacja wag
        self.weights_hidden_output += self.hidden_output.T.dot(d_output)
        self.weights_input_hidden += X.T.dot(d_hidden_layer)

    # Funkcja treningowa
    def train(self, X, y, epochs=20000):
        for _ in range(epochs):
            # Forward propagation
            output = self.forward(X)
            # Backward propagation i aktualizacja wag
            self.backward(X, y, output)



if __name__ == "__main__":      

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Tworzenie sieci neuronowej (2 wejścia, 2 ukryte neurony, 1 wyjście)
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

    nn.train(X, y, epochs=20000)

    print("Wyniki po treningu:")
    print(nn.forward(X))





