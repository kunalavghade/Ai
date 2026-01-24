import numpy as np

class NeuralNetwork:
    def __init__(self):
        # define Hypaer parameters
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # Weight Matrix
        # -- first hidden layer weight matrix
        self.wheight_1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        # -- second hidden layer weight matrix between hidden layer and output layer
        self.wheight_2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

    def sigmoid(self, z):
        return  1 / (1 + np.exp(-z))

    def forward(self, x) :
        self.z2 = np.dot(x, self.wheight_1) # multiply input with all weights of first layer
        self.a2 = self.sigmoid(self.z2) # apply sigmoid function to the result
        self.z3 = np.dot(self.a2, self.wheight_2) # multiply result of first layer with all weights of second layer
        self.a3 = self.sigmoid(self.z3) # apply sigmoid function to the result
        return self.a3
    


model = NeuralNetwork()
predicted_output = model.forward(np.array([1, 0]))
print(predicted_output)