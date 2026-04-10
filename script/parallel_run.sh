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
    local reg_factor=$2
    # Print the activation for debugging
    echo "Running Julia script with Step Size: $h, Regularization Factor: $reg_factor"

    # Run the Julia script in the background
    julia --project=. script/test_Time_Reversible_OneLayer.jl $h $reg_factor &
    
    # julia --project=. test/test_Hardcode_int.jl $h $f_abs $x_suc &
}

# Loop through the activations
for h in {0.05,0.1,0.2,0.5,1.0}; do # ,  
    for reg_factor in {0.0,1e-3,1e-5,1e-7}; do #
        run_configuration $h $reg_factor 
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
