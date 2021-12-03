#!/bin/bash

echo "Start run.sh..."

#sleep 40m
wait
# num_res_block num_filter
python trian_for_doa.py 1 16
wait
python trian_for_doa.py 1 32
wait
python trian_for_doa.py 1 64
wait
python trian_for_doa.py 1 128
wait
python trian_for_doa.py 3 8
wait
python trian_for_doa.py 3 16
wait
python trian_for_doa.py 3 32
wait
python trian_for_doa.py 3 64
wait
python trian_for_doa.py 3 128
wait
python trian_for_doa.py 1 8
wait


<< COMMENT
# num_res_block normalization
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
<< COMMENT
python trian_for_doa.py 0 sample-wise
wait
python trian_for_doa.py 1 sample-wise
wait
python trian_for_doa.py 2 sample-wise
wait
python trian_for_doa.py 3 sample-wise
wait
COMMENT


echo "End run.sh..."
