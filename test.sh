#!/bin/bash

find test/ -name '*.cpp' | xargs g++ -Wall -Wextra -Werror -pedantic -std=c++2a -I"./include" -o tests \
	&& ./tests \
	&& rm ./tests
