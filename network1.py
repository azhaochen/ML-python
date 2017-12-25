
import numpy as np
import pdb
import matplotlib.pyplot as plt

class Network():
    def __init__(self,sizes):
        self.num_layers = len(sizes)    #list as [3,4,2], numbers of neurons in each layer
        self.sizes = sizes
        self.weights = [np.random.randn(r,c) for r,c in zip(sizes[1:],sizes[:-1])]  #list, elements = layers-1, first input layer is not included
        self.biases  = [np.random.randn(r,1) for r   in sizes[1:]]                  #list, elements = layers-1, first input layer is not included
        self.costs   = [];
    
    def activiate_fun(self,z):
        ''' activiate_ function : sigmod()'''
        return 1.0/(1.0+np.exp(-z))
    
    def activiate_derivative_fun(self,z):
        ''' the derivative value of activate function at z '''
        return self.activiate_fun(z) * (1 - self.activiate_fun(z))
    
    def cost_derivative_fun(self,finalactivation,y):
        return finalactivation - y      # this is for the Least square cost function
    
    
    def feedforward(self,x):
        '''input a sample x, calculate output of network, x is column vector. '''
        output = []                     # save output of each layer
        for w,b in zip(self.weights,self.biases):
            x = self.activiate_fun(np.dot(w,x)+b)
            output.append(x);
        return x

    def delta_backprop(self,x,y):
        ''' for each sample (x,y), calculate the error(δ) of each layer, x, y must be row vectors, for list, rows of list = length of list
            gradient, the nabla (∨) weights of each neuron of each layer, 
            gradient, the nabla (∨) biases of each neuron of each layer, 
            actually, the nabla biases is equal to delta (δ), according to the equations.
        '''
        x = np.array(x);x.resize(len(x),1);                     #no matter what ,translate x to column vector
        y = np.array(y);y.resize(len(y),1);
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]      #gradient, same size of biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]     #gradient, same size of weights
        activation = x;             #current activation of one neuron, for input layer, is just the input sample
        _as = [x]                   #activations of each layer, include the input layer !! 
        _zs = [];                   #weighted input of each layer, not include the input layer !! same length of weights

        for w,b in zip(self.weights,self.biases):
            z = np.dot(w,activation)+b
            _zs.append(z)
            activation = self.activiate_fun(z)
            _as.append(activation)

        nabla_ca = self.cost_derivative_fun(_as[-1],y)              # calculate the derivative of C of a
        delta_L = nabla_ca * self.activiate_derivative_fun(_zs[-1]) # multiply element by element in the matrix
        
        nabla_b[-1] = delta_L
        nabla_w[-1] = np.dot(nabla_b[-1], _as[-1-1].T)              #attentions for this calculation !

        # calculate all the nabla for other layer neurons, last layer has calculated
        # l=1 means the last layer of neurons, l = 2 is the second-last layer,
        for l in range(2,self.num_layers):                          #range[a,b), b is not included
            delta_l = np.dot(self.weights[-l+1].T, nabla_b[-l+1]) * self.activiate_derivative_fun(_zs[-l])
            nabla_b[-l] = delta_l
            nabla_w[-l] = np.dot(nabla_b[-l], _as[-l-1].T)

        return nabla_w, nabla_b

    def minibatch_gradient(self, minibatch, eta):
        ''' calculate weights and biases gradient on small batch dataset.
            eta is the learning rate
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]      #total gradient
        nabla_w = [np.zeros(w.shape) for w in self.weights]     #total gradient
        '''
        for x,y in minibatch:
            single_w, single_b = self.delta_backprop(x,y)
            nabla_b = np.array(nabla_b) + single_b              # +  has different usage for list 
            nabla_w = np.array(nabla_w) + single_w
        # update the weights and biases
        self.weights -= nabla_w*eta / len(minibatch)
        self.biases  -= nabla_b*eta / len(minibatch)
        '''
        for x,y in minibatch:
            single_w, single_b = self.delta_backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, single_b)]    #nabla_b is list, but the element is np.array
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, single_w)]
        # update the weights and biases
        self.weights = [w-(eta/len(minibatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b-(eta/len(minibatch))*nb for b, nb in zip(self.biases, nabla_b)]

        
    def SGD(self, training_data, epochs, batch_size, eta):
        '''Train the neural network using mini-batch stochastic gradient descent. '''
        N = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            for k in range(0, N, batch_size):
                self.minibatch_gradient(training_data[k:k+batch_size], eta)
                
            print("Epoch {0} complete".format(j))
            
            


def test():
    dataset=[];

    x=np.linspace(-1,1,30);y=x;
    for i in x:
        for j in y:
            if (i**2+j**2>0.7**2) and (i**2+j**2<1):
                dataset.append([[[i],[j],[i*i]],[[1],[0]]])
            if (i**2+j**2<0.3**2):
                dataset.append([[[i],[j],[i*i]],[[0],[1]]])
    
    net = Network([3,4,2]);
    net.SGD(dataset,1700,6,0.1);
    
    plt.figure(1);plt.title('dataset,1700,6,0.1');
    for i in x:
        for j in y:
            plt.plot(i,j);
            if (i**2+j**2>0.7**2) and (i**2+j**2<1):
                tmp = net.feedforward([i,j,i*i]);
                if np.argmax(tmp)==np.argmax([[1],[0]]):
                    plt.plot(i,j,'g.');
                else:
                    plt.plot(i,j,'r.');
            if (i**2+j**2<0.3**2):
                tmp = net.feedforward([i,j,i*i]);
                if np.argmax(tmp)==np.argmax([[0],[1]]):
                    plt.plot(i,j,'g.');
                else:
                    plt.plot(i,j,'r.');

    plt.show();

    
    
test();
