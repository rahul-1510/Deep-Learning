import numpy as np

##### Library Used For Only Taking The Input From MNIST 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST data/", one_hot=True)
X_train = np.vstack([img.reshape((28*28)) for img in mnist.train.images])
Y_train = mnist.train.labels
X_test = np.vstack([img.reshape(28*28) for img in mnist.test.images])
Y_test = mnist.test.labels
del mnist
###### Taken Input

X_train = X_train[0:5000,:]

#Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#Sigmoid Derivative Function
def sigmoid_derivative(x):
    return x * (1 - x)

ip = X_train

#Initializing Output using np.ones will compare with output
op =np.ones((5000,2))

itr = 1000
lr = 0.001
iln, hln, oln = 784,10,2  #No of Input Nodes,Hidden Nodes,Output Nodes

#Initializing
h_weights = np.random.rand(iln,hln)
h_bias = np.random.rand(1,hln)
o_weights = np.random.rand(hln,oln)
o_bias = np.random.rand(1,oln)

print("Initial hidden weights: \n",end='')
print(h_weights)
print("\n")
print("Initial hidden biases: \n",end='')
print(h_bias)
print("\n")
print("Initial output weights:\n ",end='')
print(o_weights)
print("\n")
print("Initial output biases: \n",end='')
print(o_bias)

for _ in range(itr):
	#Forward Propagation
	temp1 = np.dot(ip,h_weights)
	temp1 += h_bias
	h_output = sigmoid(temp1)

	temp2 = np.dot(h_output,o_weights)
	temp2 += o_bias
	predicted_op = sigmoid(temp2)

	#Backpropagation
	error = op - predicted_op
	d_predicted_op = error * sigmoid_derivative(predicted_op)
	
	error_hidden_layer = d_predicted_op.dot(o_weights.T)

	##### Adding Sparsity Constraint to previous equation with parameters as p=0.005 and  lambda as 0.1
	d_hidden_layer = error_hidden_layer * sigmoid_derivative(h_output) + (0.1*(((-0.005)/h_weights.mean())+((1-0.005)/(h_weights.mean())))*sigmoid_derivative(h_output))

	#Updating Weights and Biases
	o_weights += h_output.T.dot(d_predicted_op) * lr
	o_bias += np.sum(d_predicted_op,axis=0,keepdims=True) * lr
	h_weights += ip.T.dot(d_hidden_layer) * lr
	h_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

	print("MSE Loss",np.square(np.subtract(op,predicted_op)).mean())

print("\n")
print("Final hidden weights: \n",end='')
print(h_weights)
print("\n")
print("Final hidden bias: \n",end='')
print(h_bias)
print("\n")
print("Final output weights:\n ",end='')
print(o_weights)
print("\n")
print("Final output bias:\n ",end='')
print(o_bias)
print("\n")

print("\nOutput from neural network after 1000 epochs: \n",end='')
print(predicted_op)


#### Got output As Follows

# [[0.99705468 0.99778285]
#  [0.99705468 0.99778285]
#  [0.99705468 0.99778285]
#  ...
#  [0.99705468 0.99778285]
#  [0.99705468 0.99778285]
#  [0.99705468 0.99778285]]
