#!/bin/bash

echo "Start test..."
python ./test1.py None
wait
python ./test2.py
wait
python ./test3.py

echo "End test..."

