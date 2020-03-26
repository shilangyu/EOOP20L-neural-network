#!/bin/bash

find tests/ -name '*.cpp' | xargs g++ -Wall -Wextra -Werror -pedantic -std=c++2a -o test \
	&& ./test \
	&& rm ./test
