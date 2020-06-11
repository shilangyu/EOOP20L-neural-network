#!/bin/bash

find src/ -name '*.cpp' | xargs g++ -D BENCH_USR_TIME -O3 -std=c++2a -I"./include" -pthread -o nn-bench-usr-time

echo 'load_train,parse_train,train,load_test,parse_test,test' > bench-usr-time-results.csv

for i in {1..20}
do
	./nn-bench-usr-time | paste -sd "," - >> bench-usr-time-results.csv
done

rm nn-bench-usr-time
