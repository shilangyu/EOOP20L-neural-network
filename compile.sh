#!/bin/bash

find src/ -name '*.cpp' | xargs g++ -O3 -std=c++2a -I"./include" -pthread -o nn
