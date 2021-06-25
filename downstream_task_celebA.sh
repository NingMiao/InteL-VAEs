#!/bin/bash
for ((i=0;i<3;i++)); do
    for ((j=0;j<8; j++)); do
        let k1=j*5
        let k2=(j+1)*5
        echo $k1 $k2
        python downstream_task_celebA.py --feature_start_id $k1 --feature_end_id $k2 &
        if [ $j = 2 ] ||[ $j = 5 ]; then
            wait
        fi
    done
    wait
done
