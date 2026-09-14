# ShallowNet (Reversible)

`ShallowNetReversible` is a time-symmetric single-hidden-layer neural variational integrator. It enforces the time-reversal symmetry `q(t) = q(T - t)` by requiring an even number of neurons and a palindromic parameter structure. Like `ShallowNet`, it uses symbolic derivatives and an OGA seed for initialization.

## Running the Parameter Scan

The parameter grid (`h`, `λ`, `f_abstol`, `x_suctol`, solver, and `dtype`) is configured at the top of `parallel_run.sh`:

```bash
# In parallel_run.sh:
INTEGRATOR="shallownet_reversible"
DP_FLAG=""             # set to "--double-pendulum" to include double pendulum

bash parallel_run.sh
```

After all jobs complete:

```bash
julia --project=scripts scripts/result_summary_shallownet_reversible.jl
```

## Harmonic Oscillator Results

### ReLU Activation

![Maximum Hamiltonian error vs timestep h (ReLU)](figures/shallownet_reversible_HO_relu_error_trend.png)

<!-- HO_RELU_TABLE_START -->

## ShallowNetReversible HO — ReLU

### h = 0.05

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td><strong>S=4, k=2, Max Error = 1.081e-02<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S4_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=2, Max Error = 1.066e-02<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S6_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=2, Max Error = 9.595e-05<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S8_k2_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 3</th><td><strong>S=4, k=3, Max Error = 5.517e-13<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S4_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=3, Max Error = 1.144e-09<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S6_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=3, Max Error = 1.175e-06<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S8_k3_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 4</th><td><strong>S=4, k=4, Max Error = 3.383e-03<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S4_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=4, Max Error = 2.775e-08<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S6_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=4, Max Error = 8.180e-08<br/><img src="figures/shallownet_reversible_HO_relu_h0.05_S8_k4_best.png" style="width:100%;min-width:180px"/></td></tr>
</tbody></table>

### h = 0.1

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td><strong>S=4, k=2, Max Error = 5.471e-03<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S4_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=2, Max Error = 1.338e-04<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S6_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=2, Max Error = 1.360e-04<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S8_k2_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 3</th><td><strong>S=4, k=3, Max Error = 1.248e-12<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S4_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=3, Max Error = 8.709e-10<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S6_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=3, Max Error = 5.151e-07<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S8_k3_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 4</th><td><strong>S=4, k=4, Max Error = 5.452e-04<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S4_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=4, Max Error = 1.295e-08<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S6_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=4, Max Error = 5.111e-08<br/><img src="figures/shallownet_reversible_HO_relu_h0.1_S8_k4_best.png" style="width:100%;min-width:180px"/></td></tr>
</tbody></table>

### h = 0.2

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td><strong>S=4, k=2, Max Error = 6.185e-04<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S4_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=2, Max Error = 5.764e-05<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S6_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=2, Max Error = 1.870e-04<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S8_k2_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 3</th><td><strong>S=4, k=3, Max Error = 7.955e-11<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S4_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=3, Max Error = 2.977e-10<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S6_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=3, Max Error = 5.466e-06<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S8_k3_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 4</th><td><strong>S=4, k=4, Max Error = 8.833e-05<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S4_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=4, Max Error = 2.985e-09<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S6_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=4, Max Error = 1.584e-06<br/><img src="figures/shallownet_reversible_HO_relu_h0.2_S8_k4_best.png" style="width:100%;min-width:180px"/></td></tr>
</tbody></table>

### h = 0.5

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td><strong>S=4, k=2, Max Error = 8.523e-04<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S4_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=2, Max Error = 5.837e-04<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S6_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=2, Max Error = 4.287e-04<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S8_k2_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 3</th><td><strong>S=4, k=3, Max Error = 1.968e-08<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S4_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=3, Max Error = 3.629e-05<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S6_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=3, Max Error = 5.273e-05<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S8_k3_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 4</th><td><strong>S=4, k=4, Max Error = 1.821e-05<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S4_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=4, Max Error = 9.556e-09<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S6_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=4, Max Error = 5.212e-06<br/><img src="figures/shallownet_reversible_HO_relu_h0.5_S8_k4_best.png" style="width:100%;min-width:180px"/></td></tr>
</tbody></table>

### h = 1.0

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td><strong>S=4, k=2, Max Error = 3.539e-03<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S4_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=2, Max Error = 3.706e-03<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S6_k2_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=2, Max Error = 1.537e-03<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S8_k2_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 3</th><td><strong>S=4, k=3, Max Error = 1.321e-06<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S4_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=3, Max Error = 4.741e-04<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S6_k3_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=3, Max Error = 1.374e-03<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S8_k3_best.png" style="width:100%;min-width:180px"/></td></tr>
<tr><th>k = 4</th><td><strong>S=4, k=4, Max Error = 3.658e-05<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S4_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, k=4, Max Error = 1.345e-07<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S6_k4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, k=4, Max Error = 8.100e-07<br/><img src="figures/shallownet_reversible_HO_relu_h1.0_S8_k4_best.png" style="width:100%;min-width:180px"/></td></tr>
</tbody></table>


<!-- HO_RELU_TABLE_END -->

### tanh Activation

![Maximum Hamiltonian error vs timestep h (tanh)](figures/shallownet_reversible_HO_tanh_error_trend.png)

<!-- HO_TANH_TABLE_START -->

## ShallowNetReversible HO — tanh

### h = 0.05

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td><strong>S=4, Max Error = 1.101e-04<br/><img src="figures/shallownet_reversible_HO_tanh_h0.05_S4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, Max Error = 3.389e-07<br/><img src="figures/shallownet_reversible_HO_tanh_h0.05_S6_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, Max Error = 1.509e-08<br/><img src="figures/shallownet_reversible_HO_tanh_h0.05_S8_best.png" style="width:100%;min-width:180px"/></td></tr></tbody></table>

### h = 0.1

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td><strong>S=4, Max Error = 1.578e-04<br/><img src="figures/shallownet_reversible_HO_tanh_h0.1_S4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, Max Error = 1.392e-06<br/><img src="figures/shallownet_reversible_HO_tanh_h0.1_S6_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, Max Error = 3.145e-07<br/><img src="figures/shallownet_reversible_HO_tanh_h0.1_S8_best.png" style="width:100%;min-width:180px"/></td></tr></tbody></table>

### h = 0.2

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td><strong>S=4, Max Error = 1.333e-04<br/><img src="figures/shallownet_reversible_HO_tanh_h0.2_S4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, Max Error = 9.554e-07<br/><img src="figures/shallownet_reversible_HO_tanh_h0.2_S6_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, Max Error = 3.048e-08<br/><img src="figures/shallownet_reversible_HO_tanh_h0.2_S8_best.png" style="width:100%;min-width:180px"/></td></tr></tbody></table>

### h = 0.5

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td><strong>S=4, Max Error = 5.079e-04<br/><img src="figures/shallownet_reversible_HO_tanh_h0.5_S4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, Max Error = 7.127e-07<br/><img src="figures/shallownet_reversible_HO_tanh_h0.5_S6_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, Max Error = 1.730e-08<br/><img src="figures/shallownet_reversible_HO_tanh_h0.5_S8_best.png" style="width:100%;min-width:180px"/></td></tr></tbody></table>

### h = 1.0

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td><strong>S=4, Max Error = 1.913e-03<br/><img src="figures/shallownet_reversible_HO_tanh_h1.0_S4_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=6, Max Error = 1.923e-05<br/><img src="figures/shallownet_reversible_HO_tanh_h1.0_S6_best.png" style="width:100%;min-width:180px"/></td><td><strong>S=8, Max Error = 2.640e-07<br/><img src="figures/shallownet_reversible_HO_tanh_h1.0_S8_best.png" style="width:100%;min-width:180px"/></td></tr></tbody></table>


<!-- HO_TANH_TABLE_END -->

## Double Pendulum Results

### ReLU Activation

![Maximum Hamiltonian error vs timestep h (ReLU)](figures/shallownet_reversible_DP_relu_error_trend.png)

<!-- DP_RELU_TABLE_START -->

## ShallowNetReversible DP — ReLU

### h = 0.05

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 3</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 4</th><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>

### h = 0.1

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 3</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 4</th><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>

### h = 0.2

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 3</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 4</th><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>

### h = 0.5

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 3</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 4</th><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>

### h = 1.0

<table>
<thead><tr><th></th><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody>
<tr><th>k = 2</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 3</th><td>—</td><td>—</td><td>—</td></tr>
<tr><th>k = 4</th><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>


<!-- DP_RELU_TABLE_END -->

### tanh Activation

![Maximum Hamiltonian error vs timestep h (tanh)](figures/shallownet_reversible_DP_tanh_error_trend.png)

<!-- DP_TANH_TABLE_START -->

## ShallowNetReversible DP — tanh

### h = 0.05

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td>—</td><td>—</td><td>—</td></tr></tbody></table>

### h = 0.1

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td>—</td><td>—</td><td>—</td></tr></tbody></table>

### h = 0.2

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td>—</td><td>—</td><td>—</td></tr></tbody></table>

### h = 0.5

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td>—</td><td>—</td><td>—</td></tr></tbody></table>

### h = 1.0

<table>
<thead><tr><th>S = 4</th><th>S = 6</th><th>S = 8</th></tr></thead>
<tbody><tr>
<td>—</td><td>—</td><td>—</td></tr></tbody></table>


<!-- DP_TANH_TABLE_END -->
