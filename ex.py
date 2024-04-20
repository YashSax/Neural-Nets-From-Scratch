from autograd_from_scratch import *
import math

module = Softmax()
y_pred = np.array([[1, 2]])
module.forward(y_pred)
print("Module out:", module.softmax_out, 1 - module.softmax_out)
print(module.backward(np.array([1, 1])))