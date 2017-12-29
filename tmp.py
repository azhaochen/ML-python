
import numpy as np
import pdb
import matplotlib.pyplot as plt
import mnist_loader
import json
import os

class Network():
    def __init__(self,sizes):
        self.num_layers = len(sizes)    #list as [3,4,2], numbers of neurons in each layer
        self.sizes = sizes
        self.weights = [np.random.randn(r,c) for r,c in zip(sizes[1:],sizes[:-1])]  #list, elements = layers-1, first input layer is not included
        self.biases  = [np.random.randn(r,1) for r   in sizes[1:]]                  #list, elements = layers-1, first input layer is not included
        self.costs   = [];
        self.valid_accuracy= [];
        self.train_accuracy= [];

    def activiate_fun(self,z):
        ''' activiate_ function : sigmod()''' 
        return 1.0/(1.0+np.exp(-z))

    def activiate_derivative_fun(self,z):
        ''' the derivative value of activate function at z '''
        return self.activiate_fun(z) * (1 - self.activiate_fun(z))

    def cost_derivative_fun(self,finalactivation,y):
        return finalactivation - y      # this is for the Least square cost function

    def calculate_cost(self,dataset):
        sum = 0;
        accuracy = 0;
        for x,y in dataset:
            output = self.feedforward(x);
            error = np.array(y)-output
            error = np.sqrt(np.dot(error.T,error)[0][0]/len(error))
            sum += error;
        return sum/len(dataset)

        
    def calculate_accuracy(self,valid_set,trans=False):
        accuracy = 0;
        for x,y in valid_set:
            output = self.feedforward(x)
            error = np.array(y)-output
            if trans and np.argmax(output)==np.argmax(y):
                accuracy +=1
            if not trans and np.argmax(output)==y:
                accuracy +=1
        return accuracy*100/len(valid_set)

        
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

    def minibatch_gradient(self, minibatch, eta, lmbda, N):
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
        self.weights = [w-(eta/len(minibatch))*nw-(eta*lmbda/N)*w for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b-(eta/len(minibatch))*nb for b, nb in zip(self.biases, nabla_b)]


    def SGD(self, training_data, epochs, batch_size, eta , lmbda , valid_set=[]):
        '''Train the neural network using mini-batch stochastic gradient descent. '''
        N = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            for k in range(0, N, batch_size):
                self.minibatch_gradient(training_data[k:k+batch_size], eta, lmbda, N)

            #calculate cost
            cost = self.calculate_cost(training_data)
            self.costs.append(cost);
            tmp = self.calculate_accuracy(training_data,True)
            self.train_accuracy.append(tmp);
            print("Epoch {0} complete, cost: {1:.7f}, accuracy: {2:.3f}".format(j,cost,tmp))
            #calculate valid_set accuracy
            if(valid_set!=[]):
                tmp = self.calculate_accuracy(valid_set,False)
                self.valid_accuracy.append(tmp);


def test():
    dataset=[];

    x=np.linspace(-1,1,40);y=x;
    for i in x:
        for j in y:
            if (i**2+j**2>0.7**2) and (i**2+j**2<1):
                dataset.append([[[i],[j],[i*i]],[[1],[0]]])
            if (i**2+j**2<0.3**2):
                dataset.append([[[i],[j],[i*i]],[[0],[1]]])

    #get the validset
    validset = []
    ndataset = []
    indexes = list(range(0,len(dataset),6))
    for i in range(0,len(dataset)):
        if i in indexes:
            validset.append(dataset[i])
        else:
            ndataset.append(dataset[i])
    dataset = ndataset

    net = Network([3,4,2]);
    epoches = 500;
    batchsize = 4
    learnrate = 0.05
    net.SGD(dataset,epoches,batchsize,learnrate,1,validset);

    plt.figure(1);
    plt.subplot(131)
    plt.title("epoch:{},batchsize:{},eta:{}".format(epoches,batchsize,learnrate));
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



    #plt.figure(2);
    plt.subplot(132)
    plt.title('costs per epoch')
    plt.plot(range(epoches),net.costs);
    plt.subplot(133)
    plt.title('validset accuracy per epoch')
    plt.plot(range(epoches),net.accuracy);
    plt.show();     #show() will block the threshold, put it at end.


def saveNet(net,filename=''):
    '''save sizes, biases and weights into file'''
    curdir = os.path.split(os.path.realpath(__file__))[0]
    file = curdir+'/net_size_w_b_randn.json';
    if filename!='':
        file = curdir+'/'+filename;
    with open(file, 'w') as f:
        data = {
            "sizes": net.sizes,
            "weights": [w.tolist() for w in net.weights],  #np.array.tolist
            "biases": [b.tolist() for b in net.biases]
            #"costs": str(net.costs.__name__)
        }
        json.dump(data,f,indent=4)
        
def initNet(filename=''):
    ''' create net from file '''
    curdir = os.path.split(os.path.realpath(__file__))[0]
    file = curdir+'/net_size_w_b_randn.json';
    if filename!='':
        file = curdir+'/'+filename;
    with open(file, 'r') as f:
        netstruct = json.load(f);
    net = Network(netstruct['sizes'])
    net.weights = [np.array(w) for w in netstruct['weights']]
    net.biases  = [np.array(b) for b in netstruct['biases']]

    return net;
    
def test2():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data);
    validation_data = list(validation_data);
    test_data = list(test_data);

    #net = Network([784, 30, 10])
    net = initNet()     #_已经提前初始化好权重和偏差
    epoches = 300;
    batchsize = 10
    learnrate = 0.05
    net.SGD(training_data[:1000],epoches,batchsize,learnrate,5,validation_data[:1000]);
    saveNet(net,'trained_regularized_20_10_0.05.json')
    plt.figure(0)
    plt.title('costs per epoch')
    plt.plot(range(epoches),net.costs);
    plt.figure(1);
    plt.title('train accuracy');
    plt.plot(range(epoches),net.train_accuracy);
    plt.figure(2);
    plt.title('valid accuracy');
    plt.plot(range(epoches),net.valid_accuracy);
    plt.show()
    
    #save network
    
        
    #pdb.set_trace()

def test3():
    net = initNet('trained_20_10_0.05.json')
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    validation_data = [z for z in validation_data]
    tmp = net.calculate_accuracy(validation_data,False)
    print(tmp)

    
test2();
