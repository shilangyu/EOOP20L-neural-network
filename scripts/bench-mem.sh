#!/bin/bash

find src/ -name '*.cpp' | xargs g++ -O3 -std=c++2a -I"./include" -pthread -o nn-bench-mem

echo 'mem[bytes]' > bench-mem-results.csv

./nn-bench-mem &

while true
do
	ps -C nn-bench-mem -o vsz= >> bench-mem-results.csv
	if [ $? -ne 0 ]; then
		break
	fi
	sleep 0.1
done

rm nn-bench-mem
