#!/bin/bash

find test/ -name '*.cpp' | xargs g++ -Wall -Wextra -Werror -pedantic -fsanitize=undefined -std=c++2a -I"./include" -o tests \
	&& ./tests \
	&& rm ./tests
