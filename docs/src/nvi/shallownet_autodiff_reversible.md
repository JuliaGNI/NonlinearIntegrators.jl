# ShallowNet (Autodiff + Reversible)

`ShallowNetAutodiffReversible` combines time-reversal symmetry with `ForwardDiff`-based derivative computation. It enforces the palindromic neuron constraint from `ShallowNetReversible` while avoiding the symbolic pre-compilation step of `ShallowNet`.

## Running the Parameter Scan

The parameter grid (`h`, `λ`, `f_abstol`, `x_suctol`, solver, and `dtype`) is configured at the top of `parallel_run.sh`:

```bash
# In parallel_run.sh:
INTEGRATOR="shallownet_autodiff_reversible"
DP_FLAG=""             # set to "--double-pendulum" to include double pendulum

bash parallel_run.sh
```

After all jobs complete:

```bash
julia --project=scripts scripts/result_summary_shallownet_autodiff_reversible.jl
```

## Harmonic Oscillator Results

### ReLU Activation

![Maximum Hamiltonian error vs timestep h (ReLU)](figures/shallownet_autodiff_reversible_HO_relu_error_trend.png)

<!-- HO_RELU_TABLE_START -->
<!-- HO_RELU_TABLE_END -->

### tanh Activation

![Maximum Hamiltonian error vs timestep h (tanh)](figures/shallownet_autodiff_reversible_HO_tanh_error_trend.png)

<!-- HO_TANH_TABLE_START -->
<!-- HO_TANH_TABLE_END -->

## Double Pendulum Results

### ReLU Activation

![Maximum Hamiltonian error vs timestep h (ReLU)](figures/shallownet_autodiff_reversible_DP_relu_error_trend.png)

<!-- DP_RELU_TABLE_START -->
<!-- DP_RELU_TABLE_END -->

### tanh Activation

![Maximum Hamiltonian error vs timestep h (tanh)](figures/shallownet_autodiff_reversible_DP_tanh_error_trend.png)

<!-- DP_TANH_TABLE_START -->
<!-- DP_TANH_TABLE_END -->
