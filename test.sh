#!/bin/bash

find tests/ -name '*.cpp' | xargs g++ -Wall -Wextra -Werror -pedantic -o test \
	&& ./test \
	&& rm ./test
