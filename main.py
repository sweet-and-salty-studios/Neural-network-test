import neural_network as nn
import numpy as np

layer_sizes = (3,5,10)
x = np.ones((layer_sizes[0], 1))

net = nn.Neural_network(layer_sizes)
prediction = net.predict(x)

print(prediction)