import numpy as np
import random

#Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#Sigmoid Derivative Function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input For XOR Gate
ip = np.array([[0,0],[0,1],[1,0],[1,1]])
#Output Of XOR Gate With Added Noise
op = np.array([[0.08],[1.07],[1.08],[0.09]])

itr = 50000
lr = 0.01
iln, hln, oln = 2,2,1  #No of Input Nodes,Hidden Nodes,Output Nodes

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
	d_predicted_op = 2*error * sigmoid_derivative(predicted_op)
	
	error_hidden_layer = d_predicted_op.dot(o_weights.T)
	d_hidden_layer = error_hidden_layer * sigmoid_derivative(h_output)

	#Updating Weights and Biases
	o_weights += h_output.T.dot(d_predicted_op) * lr
	o_bias += np.sum(d_predicted_op,axis=0,keepdims=True) * lr
	h_weights += ip.T.dot(d_hidden_layer) * lr
	h_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

	print("Mean Square Error :",np.square(np.subtract(op,predicted_op)).mean())

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

#Output
print("\nOutput For XOR Gate from neural network after 50,000 epochs: \n",end='')
print(predicted_op)
