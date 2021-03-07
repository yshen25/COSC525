import numpy as np
import sys
import matplotlib.pyplot as plt
# Shawn Shen 04132020
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""


# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self,activation, input_num, lr, weights=None):
        # print('constructor')
        self.activation = activation
        if weights is None:
            self.weights = np.random.random([input_num + 1]) # including bias
        else:
            self.weights = weights
        self.lr = lr

    #This method returns the activation of the net
    def activate(self,net):
        # print('activate')
        if self.activation == 0: # linear
            return net
        else: # logistic
            return 1/(1+np.exp(-net))

    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,Input):
        self.ins = Input
        net = np.dot(Input, self.weights)
        self.outs = round(self.activate(net), 3)
        return self.outs

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        # print('activationderivative')
        if self.activation == 0: # activation function is linear
            return 1
        else: # activation function is sigmoid
            return self.outs * (1-self.outs)
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        # print('calcpartialderivative')
        self.deltapartw = wtimesdelta * self.activationderivative() * self.ins
        wd = wtimesdelta * self.activationderivative() * self.weights[:-1]
        return wd
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        # print('updateweight')
        self.weights = self.weights - self.lr * self.deltapartw
        return 0
        
#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,numOfNeurons, activation, input_num, lr, weights=None):
        # print('constructor')
        self.d = input_num
        self.layer = []
        if weights is None:
            for i in range(numOfNeurons):
                self.layer.append(Neuron(activation, input_num, lr))
        else:
            for i in range(numOfNeurons):
                self.layer.append(Neuron(activation, input_num, lr, weights[i]))
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, Input):
        # print('calculate')
        output = []
        Input = np.append(Input, 1)
        for cell in self.layer:
            output.append(cell.calculate(Input))

        return output
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        wtimesdeltaNEW = np.zeros(self.d)
        for cell, wd in zip(self.layer, wtimesdelta):
            wtimesdeltaNEW += cell.calcpartialderivative(wd)
            cell.updateweight()

        return wtimesdeltaNEW
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights=None):
        # print('constructor')
        self.network = []
        if weights is None:
            for i in range(numOfLayers):
                self.network.append(FullyConnected(numOfNeurons[i], activation[i], inputSize, lr))
                inputSize = numOfNeurons[i]
        else:
            for i in range(numOfLayers):
                self.network.append(FullyConnected(numOfNeurons[i], activation[i], inputSize, lr, weights[i]))
                inputSize = numOfNeurons[i]

        self.loss = loss
    
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,Input):
        # print('constructor')
        for layer in self.network:
            output = layer.calculate(Input)
            #print("layer output", output)
            Input = output

        return output
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        # print('calculate')
        if self.loss == 0:# sum of squared errors
            return 0.5 * np.sum(np.square(yp-y))
        else: # binary cross entropy
            return np.sum(-(y.dot(np.log(yp)) + (np.full_like(y, 1)-y).dot(np.log(np.full_like(yp, 1)-yp))))
        
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        if self.loss == 0: # sum of squared errors
            return np.abs(y-yp)
        else: # binary cross entropy
            return -y/yp + (np.full_like(y,1)-y)/(np.full_like(yp,1)-yp)
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        yp = self.calculate(x) # predicted result
        E = self.calculateloss(yp, y) # loss
        dE = self.lossderiv(yp, y) # delta loss
        for layer in reversed(self.network):
            dE = layer.calcwdeltas(dE)
        return yp, E

def DoExample(w,x,y):
    # numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights
    model = NeuralNetwork(2, [2,2], 2, [1,1], 0, 0.5, w)
    model.train(x,y)
    for layer in model.network:
        for cell in layer.layer:
            print(cell.weights)
    return 0

def DoAnd():
    #numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights
    learning_rates = [0.9, 0.5, 0.2, 0.1]
    andgate = NeuralNetwork(1, [1], 2, [1], 1, 0.1)
    # if 0 is given, "divided by zero" error will occur. a small value is used to replace 0
    xx = np.array([[0.01,0.01], [0.01,1.01], [1.01,0.01], [1.01,1.01]])
    yy = np.array([[0.01], [0.01], [0.01], [1.01]])
    E = 1 # sum of loss for all training data
    steps = 0
    while E > 0.1:
        for x, y in zip(xx, yy):

            _, E = andgate.train(x, y)

        #print(steps, E)
        steps += 1

    for x, y in zip(xx, yy):
        yp, _ = andgate.train(x, y)
        print("x=", x, "yp=", yp)
    
    return 0

def DoXor():
    # same strategy as DoAnd()
    # numOfLayers,numOfNeurons, inputSize, activation, loss, lr, weights
    single = NeuralNetwork(1, [1], 2, [1], 1, 0.5)
    hidden = NeuralNetwork(2, [2,1], 2, [1,1], 1, 0.5)
    xx = np.array([[0.01,0.01], [0.01,1], [1,0.01], [1,1]])
    yy = np.array([[0.01], [1], [1], [0.01]])

    print("## train single perceptron")
    Etotal = 1
    steps = 0
    while (Etotal > 0.1) and (steps < 500):
        for x, y in zip(xx, yy):
            Etotal = 0
            _, E = single.train(x, y)
            Etotal += E
        print(steps, Etotal)
        steps += 1

    for x, y in zip(xx, yy):
        yp, _ = single.train(x, y)
        print("x=", x, "yp=", yp)

    print("## train network with one hidden layer")
    Etotal = 1
    steps = 0
    while Etotal > 0.1 and (steps < 500):
        for x, y in zip(xx, yy):
            Etotal = 0
            _, E = hidden.train(x, y)
            Etotal += E
        print(steps, Etotal)
        steps += 1

    for x, y in zip(xx, yy):
        yp, _ = hidden.train(x, y)
        print("x=", x, "yp=", yp)
    return 0

if __name__=="__main__":
    if (len(sys.argv)<2):
        print('a good place to test different parts of your code')
        
    elif (sys.argv[1]=='example'):
        print('run example from class (single step)')
        w = np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x = np.array([0.05,0.1])
        y = np.array([0.01,0.99])
        DoExample(w,x,y)
        
    elif(sys.argv[1]=='and'):
        print('learn and')
        DoAnd()
        
    elif(sys.argv[1]=='xor'):
        print('learn xor')
        DoXor()