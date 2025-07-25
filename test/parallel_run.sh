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
    local max_iter=$2
    local f_abs=$3
    local f_suc=$4
    # Print the activation for debugging
    echo "Running Julia script with Step Size: $h and Max Iterations: $max_iter, f_abs: $f_abs, f_suc: $f_suc"

    # Run the Julia script in the background
    julia --project=. test/test_PR.jl $h $max_iter $f_abs $f_suc &
}

# Loop through the activations
for h in {1,2,5}; do # 
    for max_iter in {100,}; do #  
        for f_abs in {"2eps()","8eps()"}; do # 
            for f_suc in {"2eps()","8eps()"}; do # 
                run_configuration $h $max_iter $f_abs $f_suc
            done
        done
    done
done

