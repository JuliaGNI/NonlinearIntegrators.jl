# ShallowNet

`ShallowNet` is the baseline single-hidden-layer neural variational integrator. It uses **symbolic derivatives** (pre-compiled via `SymbolicNeuralNetworks.jl`) to form the variational conditions, and an OGA seed (`OGA1d`) for the Newton initialization. Both `ReLUᵏ` and `tanh` activations are supported.

## Running the Parameter Scan

The parameter grid (`h`, `λ`, `f_abstol`, `x_suctol`, solver, and `dtype`) is configured at the top of `parallel_run.sh`:

```bash
# In parallel_run.sh:
INTEGRATOR="shallownet"
DP_FLAG=""             # set to "--double-pendulum" to include double pendulum

bash parallel_run.sh
```

The following results are based on the configurations in the above bash file:
```bash
# ── Configuration ─────────────────────────────────────────────────────────────
INTEGRATOR="shallownet"   # change to target integrator

DP_FLAG=""                # set to "--double-pendulum" to include DP problems
MAX_JOBS=${MAX_JOBS:-12}   # maximum number of Julia processes running simultaneously

# Neural integrator parameter grid
H_LIST="0.05 0.1 0.2 0.5 1.0" # 
REG_LIST="0.0 1e-3 1e-5 1e-7" # 
FABS_LIST="0.0" # 0.0 2.0 8.0
XSUC_LIST="2.0" # 0.0 2.0 8.0
SOLVER_LIST="backtracking" # static strongwolfe dogleg
DTYPE_LIST="Float64" #Float16 Float32
INT_TIMESPAN="100.0"
R_LIST="4 8 16"   # quadrature points
S_LIST="4 6 8"    # hidden neurons
K_LIST="3 4"    # ReLU exponent
```

After all jobs complete, generate the summary figures and error tables:

```bash
julia --project=scripts scripts/result_summary_shallownet.jl
```

Results are written to `docs/src/nvi/figures/`.



## Harmonic Oscillator Results

### ReLU Activation

![Maximum Hamiltonian error vs timestep h (ReLU)](figures/shallownet_HO_relu_error_trend.png)

<!-- HO_RELU_TABLE_START -->

=== ShallowNet HO — ReLU ===

### h=0.05
| S | k | min max Hamiltonian error |
|---|---|--------------------------|
| 4 | 2 | — |
| 4 | 3 | 5.862e-13 |
| 4 | 4 | 1.771e-02 |
| 6 | 2 | — |
| 6 | 3 | 1.769e-08 |
| 6 | 4 | 3.744e-08 |
| 8 | 2 | — |
| 8 | 3 | 1.429e-08 |
| 8 | 4 | 4.218e-07 |

**S=4, k=3** — min max Hamiltonian error: 5.862e-13

![Hamiltonian error time series (S=4, k=3, h=0.05)](figures/shallownet_HO_relu_h0.05_S4_k3_best.png)


**S=4, k=4** — min max Hamiltonian error: 1.771e-02

![Hamiltonian error time series (S=4, k=4, h=0.05)](figures/shallownet_HO_relu_h0.05_S4_k4_best.png)


**S=6, k=3** — min max Hamiltonian error: 1.769e-08

![Hamiltonian error time series (S=6, k=3, h=0.05)](figures/shallownet_HO_relu_h0.05_S6_k3_best.png)


**S=6, k=4** — min max Hamiltonian error: 3.744e-08

![Hamiltonian error time series (S=6, k=4, h=0.05)](figures/shallownet_HO_relu_h0.05_S6_k4_best.png)


**S=8, k=3** — min max Hamiltonian error: 1.429e-08

![Hamiltonian error time series (S=8, k=3, h=0.05)](figures/shallownet_HO_relu_h0.05_S8_k3_best.png)


**S=8, k=4** — min max Hamiltonian error: 4.218e-07

![Hamiltonian error time series (S=8, k=4, h=0.05)](figures/shallownet_HO_relu_h0.05_S8_k4_best.png)


### h=0.1
| S | k | min max Hamiltonian error |
|---|---|--------------------------|
| 4 | 2 | — |
| 4 | 3 | 1.246e-12 |
| 4 | 4 | 8.038e-03 |
| 6 | 2 | — |
| 6 | 3 | 2.684e-09 |
| 6 | 4 | 1.128e-06 |
| 8 | 2 | — |
| 8 | 3 | 5.997e-09 |
| 8 | 4 | 4.175e-07 |

**S=4, k=3** — min max Hamiltonian error: 1.246e-12

![Hamiltonian error time series (S=4, k=3, h=0.1)](figures/shallownet_HO_relu_h0.1_S4_k3_best.png)


**S=4, k=4** — min max Hamiltonian error: 8.038e-03

![Hamiltonian error time series (S=4, k=4, h=0.1)](figures/shallownet_HO_relu_h0.1_S4_k4_best.png)


**S=6, k=3** — min max Hamiltonian error: 2.684e-09

![Hamiltonian error time series (S=6, k=3, h=0.1)](figures/shallownet_HO_relu_h0.1_S6_k3_best.png)


**S=6, k=4** — min max Hamiltonian error: 1.128e-06

![Hamiltonian error time series (S=6, k=4, h=0.1)](figures/shallownet_HO_relu_h0.1_S6_k4_best.png)


**S=8, k=3** — min max Hamiltonian error: 5.997e-09

![Hamiltonian error time series (S=8, k=3, h=0.1)](figures/shallownet_HO_relu_h0.1_S8_k3_best.png)


**S=8, k=4** — min max Hamiltonian error: 4.175e-07

![Hamiltonian error time series (S=8, k=4, h=0.1)](figures/shallownet_HO_relu_h0.1_S8_k4_best.png)


### h=0.2
| S | k | min max Hamiltonian error |
|---|---|--------------------------|
| 4 | 2 | — |
| 4 | 3 | 7.956e-11 |
| 4 | 4 | 1.060e-03 |
| 6 | 2 | — |
| 6 | 3 | 1.713e-09 |
| 6 | 4 | 1.700e-05 |
| 8 | 2 | — |
| 8 | 3 | 2.052e-09 |
| 8 | 4 | 1.658e-06 |

**S=4, k=3** — min max Hamiltonian error: 7.956e-11

![Hamiltonian error time series (S=4, k=3, h=0.2)](figures/shallownet_HO_relu_h0.2_S4_k3_best.png)


**S=4, k=4** — min max Hamiltonian error: 1.060e-03

![Hamiltonian error time series (S=4, k=4, h=0.2)](figures/shallownet_HO_relu_h0.2_S4_k4_best.png)


**S=6, k=3** — min max Hamiltonian error: 1.713e-09

![Hamiltonian error time series (S=6, k=3, h=0.2)](figures/shallownet_HO_relu_h0.2_S6_k3_best.png)


**S=6, k=4** — min max Hamiltonian error: 1.700e-05

![Hamiltonian error time series (S=6, k=4, h=0.2)](figures/shallownet_HO_relu_h0.2_S6_k4_best.png)


**S=8, k=3** — min max Hamiltonian error: 2.052e-09

![Hamiltonian error time series (S=8, k=3, h=0.2)](figures/shallownet_HO_relu_h0.2_S8_k3_best.png)


**S=8, k=4** — min max Hamiltonian error: 1.658e-06

![Hamiltonian error time series (S=8, k=4, h=0.2)](figures/shallownet_HO_relu_h0.2_S8_k4_best.png)


### h=0.5
| S | k | min max Hamiltonian error |
|---|---|--------------------------|
| 4 | 2 | — |
| 4 | 3 | 1.968e-08 |
| 4 | 4 | 4.436e-05 |
| 6 | 2 | — |
| 6 | 3 | 1.150e-05 |
| 6 | 4 | 1.874e-05 |
| 8 | 2 | — |
| 8 | 3 | 8.064e-06 |
| 8 | 4 | 2.671e-05 |

**S=4, k=3** — min max Hamiltonian error: 1.968e-08

![Hamiltonian error time series (S=4, k=3, h=0.5)](figures/shallownet_HO_relu_h0.5_S4_k3_best.png)


**S=4, k=4** — min max Hamiltonian error: 4.436e-05

![Hamiltonian error time series (S=4, k=4, h=0.5)](figures/shallownet_HO_relu_h0.5_S4_k4_best.png)


**S=6, k=3** — min max Hamiltonian error: 1.150e-05

![Hamiltonian error time series (S=6, k=3, h=0.5)](figures/shallownet_HO_relu_h0.5_S6_k3_best.png)


**S=6, k=4** — min max Hamiltonian error: 1.874e-05

![Hamiltonian error time series (S=6, k=4, h=0.5)](figures/shallownet_HO_relu_h0.5_S6_k4_best.png)


**S=8, k=3** — min max Hamiltonian error: 8.064e-06

![Hamiltonian error time series (S=8, k=3, h=0.5)](figures/shallownet_HO_relu_h0.5_S8_k3_best.png)


**S=8, k=4** — min max Hamiltonian error: 2.671e-05

![Hamiltonian error time series (S=8, k=4, h=0.5)](figures/shallownet_HO_relu_h0.5_S8_k4_best.png)


### h=1.0
| S | k | min max Hamiltonian error |
|---|---|--------------------------|
| 4 | 2 | — |
| 4 | 3 | 1.321e-06 |
| 4 | 4 | 2.875e-05 |
| 6 | 2 | — |
| 6 | 3 | 3.232e-04 |
| 6 | 4 | 4.471e-06 |
| 8 | 2 | — |
| 8 | 3 | 5.216e-04 |
| 8 | 4 | 1.521e-05 |

**S=4, k=3** — min max Hamiltonian error: 1.321e-06

![Hamiltonian error time series (S=4, k=3, h=1.0)](figures/shallownet_HO_relu_h1.0_S4_k3_best.pdf)


**S=4, k=4** — min max Hamiltonian error: 2.875e-05

![Hamiltonian error time series (S=4, k=4, h=1.0)](figures/shallownet_HO_relu_h1.0_S4_k4_best.png)


**S=6, k=3** — min max Hamiltonian error: 3.232e-04

![Hamiltonian error time series (S=6, k=3, h=1.0)](figures/shallownet_HO_relu_h1.0_S6_k3_best.png)


**S=6, k=4** — min max Hamiltonian error: 4.471e-06

![Hamiltonian error time series (S=6, k=4, h=1.0)](figures/shallownet_HO_relu_h1.0_S6_k4_best.png)


**S=8, k=3** — min max Hamiltonian error: 5.216e-04

![Hamiltonian error time series (S=8, k=3, h=1.0)](figures/shallownet_HO_relu_h1.0_S8_k3_best.png)


**S=8, k=4** — min max Hamiltonian error: 1.521e-05

![Hamiltonian error time series (S=8, k=4, h=1.0)](figures/shallownet_HO_relu_h1.0_S8_k4_best.png)


<!-- HO_RELU_TABLE_END -->

### tanh Activation

![Maximum Hamiltonian error vs timestep h (tanh)](figures/shallownet_HO_tanh_error_trend.png)

<!-- HO_TANH_TABLE_START -->

=== ShallowNet HO — tanh ===

### h=0.05
| S | min max Hamiltonian error |
|---|--------------------------|
| 4 | 1.947e-04 |
| 6 | 1.093e-06 |
| 8 | 1.222e-08 |

**S=4** — min max Hamiltonian error: 1.947e-04

![Hamiltonian error time series (S=4, h=0.05)](figures/shallownet_HO_tanh_h0.05_S4_best.png)


**S=6** — min max Hamiltonian error: 1.093e-06

![Hamiltonian error time series (S=6, h=0.05)](figures/shallownet_HO_tanh_h0.05_S6_best.png)


**S=8** — min max Hamiltonian error: 1.222e-08

![Hamiltonian error time series (S=8, h=0.05)](figures/shallownet_HO_tanh_h0.05_S8_best.png)


### h=0.1
| S | min max Hamiltonian error |
|---|--------------------------|
| 4 | 4.881e-05 |
| 6 | 6.733e-06 |
| 8 | 1.657e-08 |

**S=4** — min max Hamiltonian error: 4.881e-05

![Hamiltonian error time series (S=4, h=0.1)](figures/shallownet_HO_tanh_h0.1_S4_best.png)


**S=6** — min max Hamiltonian error: 6.733e-06

![Hamiltonian error time series (S=6, h=0.1)](figures/shallownet_HO_tanh_h0.1_S6_best.png)


**S=8** — min max Hamiltonian error: 1.657e-08

![Hamiltonian error time series (S=8, h=0.1)](figures/shallownet_HO_tanh_h0.1_S8_best.png)


### h=0.2
| S | min max Hamiltonian error |
|---|--------------------------|
| 4 | 1.262e-05 |
| 6 | 1.664e-06 |
| 8 | 1.764e-09 |

**S=4** — min max Hamiltonian error: 1.262e-05

![Hamiltonian error time series (S=4, h=0.2)](figures/shallownet_HO_tanh_h0.2_S4_best.png)


**S=6** — min max Hamiltonian error: 1.664e-06

![Hamiltonian error time series (S=6, h=0.2)](figures/shallownet_HO_tanh_h0.2_S6_best.png)


**S=8** — min max Hamiltonian error: 1.764e-09

![Hamiltonian error time series (S=8, h=0.2)](figures/shallownet_HO_tanh_h0.2_S8_best.png)


### h=0.5
| S | min max Hamiltonian error |
|---|--------------------------|
| 4 | 1.566e-04 |
| 6 | 5.515e-07 |
| 8 | 5.900e-08 |

**S=4** — min max Hamiltonian error: 1.566e-04

![Hamiltonian error time series (S=4, h=0.5)](figures/shallownet_HO_tanh_h0.5_S4_best.png)


**S=6** — min max Hamiltonian error: 5.515e-07

![Hamiltonian error time series (S=6, h=0.5)](figures/shallownet_HO_tanh_h0.5_S6_best.png)


**S=8** — min max Hamiltonian error: 5.900e-08

![Hamiltonian error time series (S=8, h=0.5)](figures/shallownet_HO_tanh_h0.5_S8_best.png)


### h=1.0
| S | min max Hamiltonian error |
|---|--------------------------|
| 4 | 7.956e-04 |
| 6 | 2.571e-07 |
| 8 | 5.975e-08 |

**S=4** — min max Hamiltonian error: 7.956e-04

![Hamiltonian error time series (S=4, h=1.0)](figures/shallownet_HO_tanh_h1.0_S4_best.png)


**S=6** — min max Hamiltonian error: 2.571e-07

![Hamiltonian error time series (S=6, h=1.0)](figures/shallownet_HO_tanh_h1.0_S6_best.png)


**S=8** — min max Hamiltonian error: 5.975e-08

![Hamiltonian error time series (S=8, h=1.0)](figures/shallownet_HO_tanh_h1.0_S8_best.png)


<!-- HO_TANH_TABLE_END -->

## Note ##
For the Harmonic Oscillator problem, the Hamiltonian error does not always exhibit the expected oscillatory behavior, mainly due to several issues, including solver convergence problems and improperly chosen hyperparameters.

## Double Pendulum Results

### ReLU Activation

![Maximum Hamiltonian error vs timestep h (ReLU)](figures/shallownet_DP_relu_error_trend.png)

<!-- DP_RELU_TABLE_START -->
<!-- DP_RELU_TABLE_END -->


### tanh Activation

![Maximum Hamiltonian error vs timestep h (tanh)](figures/shallownet_DP_tanh_error_trend.png)

<!-- DP_TANH_TABLE_START -->
<!-- DP_TANH_TABLE_END -->

