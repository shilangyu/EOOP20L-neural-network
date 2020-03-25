#!/bin/bash

find src/ -name '*.cpp' | xargs g++ -Wall -Wextra -Werror -pedantic -o run \
	&& ./run \
	&& rm ./run
