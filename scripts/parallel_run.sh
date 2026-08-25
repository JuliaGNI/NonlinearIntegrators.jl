#!/bin/bash
# Parameter scan launcher for all neural variational integrators.
#
# Set INTEGRATOR to one of:
#   shallownet | shallownet_reversible | shallownet_autodiff |
#   shallownet_autodiff_reversible | densenet | vise
#

# ── Configuration ─────────────────────────────────────────────────────────────
INTEGRATOR="shallownet"   # change to target integrator

DP_FLAG=""                # set to "--double-pendulum" to include DP problems
MAX_JOBS=${MAX_JOBS:-12}   # maximum number of Julia processes running simultaneously

# Neural integrator parameter grid
H_LIST="0.05 0.1" # 0.2 0.5 1.0
REG_LIST="1e-3 1e-7" # 0.0  1e-5
FABS_LIST="0.0" # 2.0 8.0
XSUC_LIST="2.0" # 0.0 2.0 8.0
SOLVER_LIST="backtracking" #dogleg
DTYPE_LIST="Float64" #Float16 Float32
INT_TIMESPAN="10.0"
R_LIST="4 8 16"   # quadrature points
S_LIST="4 6 8"    # hidden neurons
K_LIST="2 3 4"    # ReLU exponent

# VISE parameter grid
VISE_R_LIST="4 8 16"
VISE_INT_TIMESPAN="1000.0"

# ── Helpers ───────────────────────────────────────────────────────────────────
# Launch a background job, blocking until a slot is free.
launch() {
    while [ "$(jobs -r -p | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 0.5
    done
    "$@" &
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case $INTEGRATOR in
  shallownet | shallownet_reversible | shallownet_autodiff | \
  shallownet_autodiff_reversible | densenet)
    SCRIPT="scripts/run_${INTEGRATOR}.jl"
    for h in $H_LIST; do
      for reg in $REG_LIST; do
        for fabs in $FABS_LIST; do
          for xsuc in $XSUC_LIST; do
            for solver in $SOLVER_LIST; do
              for dtype in $DTYPE_LIST; do
                for R in $R_LIST; do
                  for S in $S_LIST; do
                    for k in $K_LIST; do
                      echo "Launching ${INTEGRATOR} h=${h} reg=${reg} fabs=${fabs} xsuc=${xsuc} solver=${solver} dtype=${dtype} R=${R} S=${S} k=${k}"
                      launch julia --project=scripts $SCRIPT $dtype $h $reg $fabs $xsuc $INT_TIMESPAN $solver $R $S $k $DP_FLAG
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
    ;;
  # vise)
    # SCRIPT="scripts/run_vise.jl"
    # for h in $H_LIST; do
    #   for R in $VISE_R_LIST; do
    #     for dtype in $DTYPE_LIST; do
    #       echo "Launching vise h=${h} R=${R} intspan=${VISE_INT_TIMESPAN} dtype=${dtype}"
    #       julia --project=scripts $SCRIPT $h $R $VISE_INT_TIMESPAN $dtype &
    #     done
    #   done
    # done
    # ;;
  all)
    for INTEGRATOR in shallownet shallownet_reversible shallownet_autodiff \
                      shallownet_autodiff_reversible densenet vise; do
      echo "=== Starting ${INTEGRATOR} ==="
      INTEGRATOR=$INTEGRATOR bash "$0"
      echo "=== Finished ${INTEGRATOR} ==="
    done
    exit 0
    ;;
  *)
    echo "Unknown integrator: $INTEGRATOR"
    echo "Choose from: shallownet, shallownet_reversible, shallownet_autodiff,"
    echo "             shallownet_autodiff_reversible, densenet, vise, all"
    exit 1
    ;;
esac

wait
echo "All jobs for ${INTEGRATOR} completed."
