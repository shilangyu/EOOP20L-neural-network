---
title: EOOP 20L Preliminary project - Neural Network
author:
  - Marcin Wojnarowski (303880)
published: true
urlcolor: "cyan"
---

_Disclaimer_: This document is not meant to be an explanation of the whats and hows of a Neural Network. It will strictly focus on the interface and implementation.

## 1. Description

This project will focus on implementing a basic multilayer perceptron (known as a ANN: Artificial Neural Network) stripped of all of the more advanced concepts like normalizing weights, mini-batch training, or vectorized computation. The implementation will consist of 5 classes in total, each handling a separate part of what a Neural Network forms.

As a Proof of Concept I will train a NN to solve the [XOR](https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b) problem: it is known as a simple non-linear problem that requires at least one hidden layer in order to be solved, thus solving it proves that a given NN is able to solve other arbitrary non-linear problems. More or less effectively, but still. If time allows, the NN will be also put against the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database).

### Classes

#### Serializer

This class only exists as a base of a different one. It requires that deriving classes have implemented a serialize method and a static deserialize method. Then, the `Serializer` is able to provide such methods as `from_file` or `to_file`, allowing for easy serialization handling no matter who is the parent class.

This class will prove to be very useful when saving the NN's weights to a file and then loading them back in whenever needed. This allows for an interruptible training cycle. Another thing is the ability to configure a NN from a text file rather than fiddling with the code.

#### Config

`Config` conforms `Serializable`[^1]. Stores sizes of the network: input, output, hidden layers, hidden neurons.

#### Matrix

The main purpose of the `Matrix` class is to simplify matrix operations. It will override the 4 basic arithmetic operators `*/-+` providing a layer of abstraction when using 2d arrays. `*-+` will work for both scalars and other matrices while `/` will be constrained to just scalars. Additionally there will be 2 other linear algebraic operations: transposition and the dot product. Cross product will be omitted on purpose: while it would be great for completeness, the usefulness is minimal.

`Matrix` conforms `Serializable` as well.

#### NNFunctions

`NNFunctions` stores function used by the NN: activation function, derivative of the activation function, output layer function and cost function. Constructing a `NNFunction` object consists of providing the previously mentioned function, or choosing ready function from an enum.

#### NeuralNetwork

This is the _brain_ class. Uses all of the classes above to construct a friendly interface for training and performing guesses.

---

- The code will be formatted using `clang-format` with the `Google` preset
- Compiled with `g++` version 10 with the `-std=c++2a` flag
- Naming convention:
  - Type aliases, classes, structs, enums, concepts: PascalCase
  - Private fields: snake_case with an underscore at the end
  - All the rest: snake_case

[^1]: Serializable: a c++20 concept describing the constraints given by the `Serializer` class
