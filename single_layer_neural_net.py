from numpy import random, dot, array, exp


# define sigmoid function
def sigmoid(x):
        return 1 / (1 + exp(-x))


# derivative of sigmoid function to find local minima of loss function
def sigmoid_derivative(x):
        return x * (1 - x)


# adjust weights on each iteration
def minimize(training_inputs, output, weights, error):
        adjustment = dot(training_inputs.T, error * sigmoid_derivative(output))
        weights += adjustment
        return weights


# feedforward and backpropagate on each iteration for 10k iterations
def train(training_inputs, training_outputs, weights):
        for i in range(10000):
                # z = summation(xi.wi) + bias (bias not considered here)
                # output = sigmoid(z)
                output = sigmoid(dot(training_inputs, weights))
                
                # error = actual - predicted
                error = training_outputs - output
                
                # new weights
                weights = minimize(training_inputs, output, weights, error)

        return weights


# predict for custom inputs
def predict(inputs, weights):
        return sigmoid(dot(inputs, weights))


if __name__ == '__main__':
        # init training and testing data
        training_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
        training_outputs = array([[0, 1, 1, 0]]).T
        
        # so that every time, the same random numbers are generated
        random.seed(1)
        
        # initialise random weights
        weights = 2 * random.random((3, 1)) - 1
        
        # get final weights after training
        weights = train(training_inputs, training_outputs, weights)
        output = predict(array(list(map(int, input().split()))), weights)

        print(output)