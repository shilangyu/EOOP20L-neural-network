#!/bin/bash

echo downloading training data...
curl https://pjreddie.com/media/files/mnist_train.csv -O
echo downloading testing data...
curl https://pjreddie.com/media/files/mnist_test.csv -O
