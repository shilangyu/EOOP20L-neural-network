#!/bin/bash

find src/ -name '*.cpp' | xargs g++ -D BENCH_USR_TIME -O3 -std=c++2a -I"./include" -pthread -o nn-bench-usr-time

echo 'load_train[s],parse_train[s],train[s],load_test[s],parse_test[s],test[s]' > bench-usr-time-results.csv

for i in {1..20}
do
	./nn-bench-usr-time | paste -sd "," - >> bench-usr-time-results.csv
done

rm nn-bench-usr-time
