#!/bin/bash

find . -name '*.cpp' -not -path './src/main.cpp' | xargs g++ -Wall -Wextra -Werror -pedantic -fsanitize=undefined -std=c++2a -I"./include" -pthread -o tests \
	&& ./tests \
	&& rm ./tests
