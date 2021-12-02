#!/bin/bash

echo "Start run.sh..."
python trian_for_doa.py 0 None
wait
python trian_for_doa.py 1 None
wait
python trian_for_doa.py 2 None
wait
python trian_for_doa.py 3 None
wait
python trian_for_doa.py 4 None
wait
python trian_for_doa.py 5 None
wait
python trian_for_doa.py 0 whole
wait
python trian_for_doa.py 1 whole
wait
python trian_for_doa.py 2 whole
wait
python trian_for_doa.py 3 whole
wait
python trian_for_doa.py 4 whole
wait
python trian_for_doa.py 5 whole
wait
#python trian_for_doa.py 0 sample-wise
wait
#python trian_for_doa.py 1 sample-wise
wait
#python trian_for_doa.py 2 sample-wise
wait
#python trian_for_doa.py 3 sample-wise
wait

echo "End run.sh..."
