#! bin/usr/env python3
import numpy as np

class Neuron():
    # each neuron accepts 1d numpy array as input, regardless of layer type
    # output is a single number
    # no need to append 1 to the end of input to use bias
    # bias is claculatied seperately
    def __init__(self, acfunc, InputNum, lr, weights=None):
        if acfunc in [0,1]:
            self.acfunc = acfunc # activation function
        else:
            raise ValueError(f"activation function type wrong: {acfunc}")
        #self.d = InputShape # dimension of input (2d tuple)
        self.lr = lr
        self.bias = np.random.rand()
        if weights is None:
            self.weights = np.random.random(InputNum)
        else:
            if not weights.shape == InputNum:
                raise ValueError(f"weights shape ({weights[0].shape}) dosen't match input ({InputShape.shape})")
            else:
                self.weights = weights

    def activate(self, net):
        if self.acfunc == 0: # linear
            return net
        else: # logistic
            return 1/(1+np.exp(-net))

    def activationderivative(self):
        if self.acfunc == 0: # activation function is linear
            return 1
        else: # activation function is sigmoid
            return self.outs * (1-self.outs)

    def calculate(self, Input):
        self.ins = Input # record last input for back-propagation
        net = np.dot(Input, self.weights) + self.bias
        self.outs = self.activate(net)
        return self.outs
        
    def calcpartialderivative(self, wtimesdelta):
        
        self.deltapartw = wtimesdelta * self.activationderivative() * self.ins
        wd = wtimesdelta * self.activationderivative() * self.weights
        return wd
    
    def updateweight(self):
        self.weights = self.weights - self.lr * self.deltapartw
        return 0

    def updateweight_convolution(self, newweight, newbias):
        # update from upper level, used in convolutional layer
        self.weights = newweight
        self.bias = newbias
        return 0

class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self,LayerShape, acfunc, InputShape, lr, weights=None):

        self.InputShape = InputShape
        self.layer = []
        if weights is None:
            for i in range(LayerShape):
                self.layer.append(Neuron(acfunc, InputShape, lr))
        else:
            for i in range(LayerShape):
                self.layer.append(Neuron(acfunc, InputShape, lr, weights[i]))
        
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, Input):

        output = []
        Input = np.append(Input, 1)
        for cell in self.layer:
            output.append(cell.calculate(Input))

        return output
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta):
        wtimesdeltaNEW = np.zeros(self.InputShape)
        for cell, wd in zip(self.layer, wtimesdelta):
            wtimesdeltaNEW += cell.calcpartialderivative(wd)
            cell.updateweight()

        return wtimesdeltaNEW

class convolutionalLayer:
    def __init__(self, KernelShape:int, KernelNum:int, acfunc:int, InputShape:int, lr:float, weights=None):
        # the KernelShape represents side lenght of square
        self.InputShape = InputShape
        self.OutputShape = InputShape-KernelShape+1
        # each layer is seperated
        self.layer = []
        for i in range(KernelNum):
            self.layer.append([])

        with np.nditer(self.layer, op_flags=['writeonly'], flags=["refs_ok"]) as it:
            for slot in it:
                slot[...] = Neuron(acfunc, (InputShape[0], InputShape[1], numofkernels), lr, weights=weights)

    def calculate(self, Input):
        output = np.zeros(self.OutputShape)
        for row_idx in range(self.OutputShape[0]):
            for col_idx in range(self.OutputShape[1]):
                self.layer[row_idx,col_idx].calcualte()