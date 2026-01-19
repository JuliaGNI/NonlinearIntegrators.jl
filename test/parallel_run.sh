#!/bin/bash

# export JULIA_NUM_THREADS=20
# for i in {1..10}
# do
#     julia IntegratorNN/test_sh.jl "$i"|| true
# done

# seq 12 | xargs -I{} -P 6 bash -c 'julia IntegratorNN/test_sh2.jl {}|| true'


# Function to run the Julia script with the specified activation function
run_configuration() {
    local h=$1
    local f_abs=$2
    local x_abs=$3
    # Print the activation for debugging
    echo "Running Julia script with Step Size: $h, f_abs: $f_abs, x_abs: $x_abs"

    # Run the Julia script in the background
    julia --project=. test/test_Time_Reversible_OneLayer.jl $h $f_abs $x_abs &
}

# Loop through the activations
for h in {0.1,0.2,0.5,1.0,2.0,5.0}; do # ,  
    for f_abs in {2.0,8.0}; do # 
        for x_abs in {2.0,8.0}; do # 
            run_configuration $h $f_abs $x_abs
        done
    done
done

