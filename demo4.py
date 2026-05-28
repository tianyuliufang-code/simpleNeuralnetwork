import numpy as np
import matplotlib.pyplot as pyplot
import scipy.special



class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        pass

    def train(self,input_lists,target_list):
        inputs = np.array(input_lists,ndmin=2).T
        targets = np.array(target_list,ndmin=2).T
        
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)


        outputs_errors = targets-final_outputs
        hidden_errors = np.dot(self.who.T,outputs_errors)

        self.who += self.lr * np.dot(
            (outputs_errors * final_outputs * (1.0 - final_outputs)),
            np.transpose(hidden_outputs),
        )

        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs),
        )

        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list,ndmin=2).T

        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# # import numpy as np
# print(np.random.rand(3,3) -0.5)

# print(n.query([1.0,0.5,-1.5]))

training_data_file = open("mnist_dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5

for e in range(epochs):
    print("epoch", e + 1)
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
    pass


test_data_file = open("mnist_dataset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# all_values = test_data_list[0].split(',')
# print(all_values)


# image_array = np.asarray(all_values[1:], dtype=float).reshape((28, 28))
# pyplot.imshow(image_array, cmap='Greys', interpolation='None') 
# pyplot.show()
# pyplot.savefig("mnist_preview.png")
# pyplot.close()

# print(n.query((np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01))

#test the neural network


scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    # print(correct_label,"correct label")
    inputs = (np.asarray(all_values[1:], dtype=float)/255.0*0.99)+0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if(label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
# print(scorecard)
print("performance = ",scorecard_array.sum() / scorecard_array.size)
