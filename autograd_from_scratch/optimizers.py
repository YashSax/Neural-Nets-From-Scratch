import numpy as np
from abc import ABC, abstractmethod
from typing import List
from .modules import *

class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, lr: float):
        self.lr = lr
    
    def add_layers(self, modules: List[Module]):
        self.layers = [mod for mod in modules if type(mod) == Layer]

    def step(self):
        for layer in self.layers:
            mean_grad = np.transpose(np.mean(layer.grad, axis=0))
            layer.weights -= self.lr * mean_grad


class Adam(Optimizer):
    def __init__(self, lr: float, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.ms = {}
        self.vs = {}
        self.epsilon = 1e-8
    
    def add_layers(self, modules: List[Module]):
        self.layers = [mod for mod in modules if type(mod) == Layer]
        for layer in self.layers:
            self.ms[layer] = np.zeros_like(layer.weights)
            self.vs[layer] = np.zeros_like(layer.weights)
    
    def step(self):
        for layer in self.layers:
            mean_grad = np.transpose(np.mean(layer.grad, axis=0))
            self.ms[layer] = self.beta1 * self.ms[layer] + (1 - self.beta1) * mean_grad
            self.vs[layer] = self.beta2 * self.vs[layer] + (1 - self.beta2) * (mean_grad ** 2)
            layer.weights -= self.lr * self.ms[layer] / np.sqrt(self.vs[layer] + self.epsilon)
