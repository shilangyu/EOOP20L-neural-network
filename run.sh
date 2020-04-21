#!/bin/bash

find src/ -name '*.cpp' | xargs g++ -Wall -Wextra -Werror -pedantic -fsanitize=undefined -std=c++2a -I"./include" -o run \
	&& ./run \
	&& rm ./run
