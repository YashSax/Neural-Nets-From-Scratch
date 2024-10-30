# Neural-Nets-From-Scratch

`neural_nets_from_scratch` is a neural network library I wrote as an effort to better understand how libraries like PyTorch work under the hood. `neural_nets_from_scratch` can be used to create Deep Neural Networks of arbitrary sizes. As this project is mostly for my learning, I've only implemented linear layers, two types of activation functions, and a couple different optimizers. P.S: You might see cases where I refer to this project as an autograd library: since there isn't a dynamic computation graph or anything of the sort (gradient calculation only happens on the specific `Modules` where I write both a `forward` and `backward` pass explicitly), I suppose this project can't really be classified as a true autograd.

![classification](https://github.com/YashSax/Neural-Nets-From-Scratch/assets/46911428/244f4863-8dd1-4a1c-836d-15d47f8c742d)
![function_approximation](https://github.com/YashSax/Neural-Nets-From-Scratch/assets/46911428/867d4e0b-0dd2-4194-8398-b22ab1759b48)
![linear_regression](https://github.com/YashSax/Neural-Nets-From-Scratch/assets/46911428/d506a631-8170-453a-8342-490d666b444b)


Example Usage:

```python3
from autograd_from_scratch import *

model = Sequential([
    Layer(1, 20),
    ReLU(),
    Layer(20, 20),
    Sigmoid(),
    Layer(20, 20),
    Sigmoid(),
    Layer(20, 1)
],
criterion=MSELoss(),
optimizer=Adam(lr=0.005, beta1=0.9, beta2=0.99))

def generate_training_example(batch_size):
    model_in = np.random.randint(low=-2*np.pi, high=2*np.pi, size=(batch_size, 1))
    model_out = np.sin(model_in)
    return x, y

for i in range(100):
 x, y = generate_training_example(32)
 pred_out = model.forward(x)
 loss = model.calculate_loss(y, pred_out)
 model.backward()
 model.optimizer.step()
```

Currently it supports:

* Layers: Linear

* Activation Functions: ReLU, Sigmoid
* Optimizers: Adam, SGD

Some example Jupyter notebooks showing functionality:
 - `ex1_linear_regression.ipynb` : Simple linear regression with a tiny neural network
 - `ex2_function_approximation.ipynb` : Approximation of $sin(x)$ using a deep neural network and Adam
 - `ex3_classification.ipynb` : Learning an image via classification as points as black or white given (x, y) coordinates
