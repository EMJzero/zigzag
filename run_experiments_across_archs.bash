#!/bin/bash

# Define the two lists
mains=("main_salsa" "main")
# mains=("main")
list1=("I" "II" "III" "IV" "V" "VI" "VII" "VIII" "IX" "X" "XI" "XII" "XIII" "XIV" "XV" "XVI" "XVII")
# list1=("I" "II" "III" "IV" "V" "VI" "VII" "VIII" "IX" "X" "XI" "XII" "XIII" "XIV" "XV")
list2=("eyeriss" "simba" "ff_tpu")
# list2=("ff_tpu")

# Iterate over both lists
for main in "${mains[@]}"; do
    for val1 in "${list1[@]}"; do
        for val2 in "${list2[@]}"; do
            echo Handling ${val2} ${val1} with ${main}

            { time_output=$( { time python3.11 ${main}.py \
                --mapping zigzag/inputs/mapping/${val2}_like_conv.yaml \
                --model emjzero/${val1}.yaml \
                --accelerator zigzag/inputs/hardware/${val2}_like.yaml; } 2>&1 | tee /dev/tty ); } 2>&1

            mkdir emjzero/results/${val2}_${val1}
            python3.11 visualization.py | tee emjzero/results/${val2}_${val1}/${main}.txt

            # Extract the real time from the time command output and convert to pure seconds
            real_time=$(echo "$time_output" | grep "real" | awk '{split($2, a, "m"); print a[1] * 60 + substr(a[2], 1, length(a[2])-1) "s"}')

            # Append the extracted time to the results file and display it
            echo "Execution time: $real_time" | tee -a emjzero/results/${val2}_${val1}/${main}.txt
        done
    done
done