#pragma once

// nnpp is a simple neural network library written in C++ that provides basic
// functionality for creating and training neural networks. It is designed to be
// easy to use and understand, making it suitable for educational purposes and
// small projects.

#include <vector>

class NN
{

public:
    NN();
    ~NN();
    void train();

private:
};

class Layer
{
public:
    Layer();
    ~Layer();
    void forward();

private:
};

class Neuron
{

public:
    Neuron();
    ~Neuron();
    void activate();

private:
};
