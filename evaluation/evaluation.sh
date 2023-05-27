# !/bin/bash

for D in 'test' 'swap_s' 'swap_a' 'swap_n' 'wider'
do
    python -u leaveout_ASP.py \
    --test_type $D
done
