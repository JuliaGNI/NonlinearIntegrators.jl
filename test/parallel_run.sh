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
    local x_suc=$3
    # Print the activation for debugging
    echo "Running Julia script with Step Size: $h, f_abs: $f_abs, x_suc: $x_suc"

    # Run the Julia script in the background
    julia --project=. test/test_NonLinear_OneLayer_GML.jl $h $f_abs $x_suc &
    
    # julia --project=. test/test_Hardcode_int.jl $h $f_abs $x_suc &
}

# Loop through the activations
for h in {0.1,0.2,0.5,1.0}; do # ,  
    for f_abs in {2.0,8.0}; do #
        for x_suc in {2.0,8.0}; do #
            run_configuration $h $f_abs $x_suc 
        done
    done
done

# MAX_JOBS=24

# run_with_limit() {
#     while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
#         wait -n
#     done
#     "$@" &
# }

# for h in {0.1,0.2,0.5,1.0}; do # ,  
#     for f_abs in {2.0,8.0}; do #
#         for x_suc in {2.0,8.0}; do #
#             run_with_limit julia --project=. test/test_NonLinear_OneLayer_GML.jl $h $f_abs $x_suc  &
#         # run_with_limit julia --project=. test/test_Hardcode_int.jl $h $f_abs $x_suc
#         done
#     done
# done

# wait
