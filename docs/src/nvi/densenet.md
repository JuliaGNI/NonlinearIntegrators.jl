# DenseNet

`DenseNet` is a three-hidden-layer dense neural variational integrator. It uses an LSGD (Least-Squares Gradient Descent) training step for initialization rather than an OGA seed, giving it greater expressiveness at the cost of a more expensive initialization phase.

## Running the Parameter Scan

The parameter grid (`h`, `λ`, `f_abstol`, `x_suctol`, solver, and `dtype`) is configured at the top of `parallel_run.sh`:

```bash
# In parallel_run.sh:
INTEGRATOR="densenet"
DP_FLAG=""             # set to "--double-pendulum" to include double pendulum

bash parallel_run.sh
```

After all jobs complete:

```bash
julia --project=scripts scripts/result_summary_densenet.jl
```

## Harmonic Oscillator Results

### ReLU Activation

![Maximum Hamiltonian error vs timestep h (ReLU)](figures/densenet_HO_relu_error_trend.png)

<!-- HO_RELU_TABLE_START -->
<!-- HO_RELU_TABLE_END -->

### tanh Activation

![Maximum Hamiltonian error vs timestep h (tanh)](figures/densenet_HO_tanh_error_trend.png)

<!-- HO_TANH_TABLE_START -->
<!-- HO_TANH_TABLE_END -->

## Double Pendulum Results

### ReLU Activation

![Maximum Hamiltonian error vs timestep h (ReLU)](figures/densenet_DP_relu_error_trend.png)

<!-- DP_RELU_TABLE_START -->
<!-- DP_RELU_TABLE_END -->

### tanh Activation

![Maximum Hamiltonian error vs timestep h (tanh)](figures/densenet_DP_tanh_error_trend.png)

<!-- DP_TANH_TABLE_START -->
<!-- DP_TANH_TABLE_END -->
