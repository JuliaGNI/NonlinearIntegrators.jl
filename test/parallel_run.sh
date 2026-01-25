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
    local k_relu=$2
    # Print the activation for debugging
    echo "Running Julia script with Step Size: $h, k_relu: $k_relu"

    # Run the Julia script in the background
    julia --project=. test/test_Hardcode_int.jl $h $k_relu &
}

# Loop through the activations
for h in {0.1,0.2,0.5,1.0,2.0}; do # ,  
    for k in {2,3,4}; do # 
        run_configuration $h $k 
    done
done

