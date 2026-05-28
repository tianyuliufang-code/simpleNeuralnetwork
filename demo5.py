import numpy as np
import matplotlib.pyplot as pyplot
# %matplotlib inline

data_file = open("mnist_dataset/mnist_train_100.csv", "r")
data_list = data_file.readlines()
data_file.close()

print(len(data_list))
print(data_list[0])

all_values = data_list[1].split(',')
image_array = np.asarray(all_values[1:], dtype=float).reshape((28, 28))
pyplot.imshow(image_array, cmap='Greys', interpolation='None')
pyplot.show()
