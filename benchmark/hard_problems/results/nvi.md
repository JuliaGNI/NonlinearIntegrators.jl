# Networks of the package on the hard problems (phase 2)

N1 = ShallowNet(ReLU³, S = 4, R = 4) with the kink-free bias interval [1.1, π]; N2 = the same
with [-π, π], and ShallowNet(tanh) with S = R = 4 and S = R = 8. DogLeg, regularisation 1e-5,
≤ 1000 iterations per step. Step classes are counted per step and degree of freedom. "Linear
front at that time": the smallest q error of an accurate linear run that took at most as long.

## P1_eps0.001

| network | h Ω | status | unconv. steps | max residual | q error | J | wall time [s] | P₃-equiv / spline / degen | diff to CGVI(P₃) | linear front at that time |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 ReLU3 kinkfree S=4 | 0.1 | ok | 0 | 1.95e-13 | 1.38e-07 | 4.24e-04 | 266 | 62831 / 0 / 0 | 3.34e-12 | 1.75e-13 |
| N2 ReLU3 S=4 | 0.1 | unconverged | 4 | 6.02e-06 | 2.47e-05 | 4.72e-04 | 266 | 62827 / 4 / 0 | 2.47e-05 | 1.75e-13 |
| N2 tanh S=4 | 0.1 | unconverged | 62774 | 9.14e-04 | 3.23e-01 | 5.76e-01 | 970 | — | 3.23e-01 | 1.75e-13 |

- N1 ReLU3 kinkfree S=4: 1 runs, 0 not converged (0 %), 0 better than the linear front by 3×
- N2 ReLU3 S=4: 1 runs, 1 not converged (100 %), 0 better than the linear front by 3×
- N2 tanh S=4: 1 runs, 1 not converged (100 %), 0 better than the linear front by 3×
- N1 with every step P₃-equivalent and converged: 1 runs, max diff to CGVI(P₃) 3.34e-12

## P1_eps0.01

| network | h Ω | status | unconv. steps | max residual | q error | J | wall time [s] | P₃-equiv / spline / degen | diff to CGVI(P₃) | linear front at that time |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 ReLU3 kinkfree S=4 | 0.1 | ok | 0 | 1.47e-13 | 1.38e-08 | 4.26e-03 | 26.5 | 6283 / 0 / 0 | 2.78e-13 | 1.01e-13 |
| N2 ReLU3 S=4 | 0.1 | ok | 0 | 1.46e-13 | 1.38e-08 | 4.26e-03 | 26.6 | 6283 / 0 / 0 | 3.60e-13 | 1.01e-13 |
| N2 tanh S=4 | 0.1 | unconverged | 6273 | 6.56e-04 | 3.66e-02 | 9.10e-02 | 93.9 | — | 3.66e-02 | 1.01e-13 |
| N2 tanh S=8 | 0.1 | unconverged | 5734 | 1.67e-03 | 1.35e-02 | 3.14e-02 | 209 | — | 1.35e-02 | 1.01e-13 |
| N1 ReLU3 kinkfree S=4 | 0.3 | unconverged | 138 | 4.35e-03 | 1.68e-02 | 4.23e-02 | 12.6 | 2093 / 0 / 1 | 1.68e-02 | 1.01e-13 |
| N2 ReLU3 S=4 | 0.3 | unconverged | 143 | 8.40e-03 | 1.39e-02 | 3.57e-02 | 12.8 | 2093 / 0 / 1 | 1.39e-02 | 1.01e-13 |
| N2 tanh S=4 | 0.3 | unconverged | 2088 | 1.95e-03 | 3.95e-01 | 5.71e-01 | 32 | — | 3.95e-01 | 1.01e-13 |
| N2 tanh S=8 | 0.3 | unconverged | 2066 | 2.24e-03 | 5.68e-01 | 1.36e+00 | 73.1 | — | 5.68e-01 | 1.01e-13 |
| N1 ReLU3 kinkfree S=4 | 1 | unconverged | 222 | 9.04e-02 | 3.32e-01 | 1.38e-01 | 5.02 | 583 / 45 / 0 | 3.20e-01 | 1.01e-13 |
| N2 ReLU3 S=4 | 1 | unconverged | 210 | 5.75e-02 | 1.74e-01 | 1.08e-01 | 5.06 | 578 / 50 / 0 | 1.62e-01 | 1.01e-13 |
| N2 tanh S=4 | 1 | unconverged | 613 | 3.99e-03 | 9.95e-01 | 9.53e-01 | 10.2 | — | 9.90e-01 | 1.01e-13 |
| N2 tanh S=8 | 1 | unconverged | 626 | 8.11e-03 | 1.07e+00 | 9.99e-01 | 20.5 | — | 1.07e+00 | 1.01e-13 |
| N1 ReLU3 kinkfree S=4 | 3 | unconverged | 76 | 4.03e+00 | 9.75e-01 | 9.96e-01 | 2.01 | 178 / 31 / 0 | 9.41e-01 | 1.01e-13 |
| N2 ReLU3 S=4 | 3 | unconverged | 78 | 2.28e+00 | 1.34e+00 | 1.31e+00 | 1.81 | 174 / 35 / 0 | 1.18e+00 | 1.01e-13 |
| N2 tanh S=4 | 3 | unconverged | 209 | 5.60e+02 | 6.56e+01 | 5.44e+03 | 3.74 | — | 6.60e+01 | 1.01e-13 |
| N2 tanh S=8 | 3 | unconverged | 209 | 7.99e+00 | 4.97e+00 | 3.52e+01 | 8.85 | — | 6.05e+00 | 1.01e-13 |
| N1 ReLU3 kinkfree S=4 | 10 | unconverged | 56 | 2.71e+06 | 1.64e+03 | 3.73e+06 | 0.763 | 46 / 14 / 2 | 1.00e+00 | 1.01e-13 |
| N2 ReLU3 S=4 | 10 | unconverged | 58 | 2.22e+06 | 2.34e+03 | 7.24e+06 | 0.718 | 42 / 15 / 5 | 1.00e+00 | 1.01e-13 |
| N2 tanh S=4 | 10 | unconverged | 62 | 6.38e+10 | 1.75e+05 | 5.47e+10 | 0.805 | — | 1.00e+00 | 1.01e-13 |
| N2 tanh S=8 | 10 | unconverged | 62 | 1.08e+02 | 1.01e+00 | 1.00e+00 | 1.68 | — | 1.00e+00 | 1.01e-13 |

- N1 ReLU3 kinkfree S=4: 5 runs, 4 not converged (80 %), 0 better than the linear front by 3×
- N2 ReLU3 S=4: 5 runs, 4 not converged (80 %), 0 better than the linear front by 3×
- N2 tanh S=4: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N2 tanh S=8: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N1 with every step P₃-equivalent and converged: 1 runs, max diff to CGVI(P₃) 2.78e-13

## P3_e0.99

| network | h Ω | status | unconv. steps | max residual | q error | H | L | ϖ | wall time [s] | P₃-equiv / spline / degen | diff to CGVI(P₃) | linear front at that time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N1 ReLU3 kinkfree S=4 | 0.1 | unconverged | 233 | 1.58e-09 | 3.69e+02 | 1.38e+02 | 6.98e-11 | 5.27e-01 | 5.12 | 1256 / 0 / 0 | 1.59e-14 | — |
| N2 ReLU3 S=4 | 0.1 | unconverged | 251 | 1.45e-09 | 3.69e+02 | 1.38e+02 | 5.32e-11 | 5.27e-01 | 5.04 | 1256 / 0 / 0 | 1.11e-14 | — |
| N2 tanh S=4 | 0.1 | unconverged | 628 | 4.64e-02 | 3.67e+02 | 1.37e+02 | 7.89e+00 | 5.08e-01 | 23.1 | — | 6.81e-03 | — |
| N2 tanh S=8 | 0.1 | unconverged | 628 | 1.92e-02 | 3.55e+02 | 1.28e+02 | 8.29e-01 | 2.63e-01 | 61.6 | — | 4.89e-02 | — |
| N1 ReLU3 kinkfree S=4 | 0.3 | unconverged | 38 | 3.11e-10 | 4.22e+02 | 1.80e+02 | 6.10e-13 | 4.85e-01 | 1.36 | 418 / 0 / 0 | 5.01e-15 | — |
| N2 ReLU3 S=4 | 0.3 | unconverged | 39 | 4.47e-10 | 4.22e+02 | 1.80e+02 | 2.86e-12 | 4.85e-01 | 1.42 | 418 / 0 / 0 | 6.10e-15 | — |
| N2 tanh S=4 | 0.3 | unconverged | 209 | 1.54e-02 | 4.21e+02 | 1.79e+02 | 1.63e-01 | 4.77e-01 | 5.78 | — | 2.84e-03 | — |
| N2 tanh S=8 | 0.3 | unconverged | 209 | 6.98e-02 | 3.53e+02 | 1.27e+02 | 2.07e-03 | 5.37e-01 | 19.2 | — | 1.63e-01 | — |
| N1 ReLU3 kinkfree S=4 | 1 | ok | 0 | 7.28e-11 | 4.33e+02 | 1.94e+02 | 1.85e-14 | 4.72e-01 | 0.379 | 124 / 0 / 0 | 1.19e-15 | — |
| N2 ReLU3 S=4 | 1 | unconverged | 1 | 1.26e-10 | 4.33e+02 | 1.94e+02 | 1.56e-13 | 4.72e-01 | 0.376 | 124 / 0 / 0 | 3.69e-15 | — |
| N2 tanh S=4 | 1 | unconverged | 62 | 9.58e-03 | 4.33e+02 | 1.94e+02 | 3.63e-03 | 4.72e-01 | 1.44 | — | 1.05e-04 | — |
| N2 tanh S=8 | 1 | unconverged | 62 | 2.70e-02 | 4.16e+02 | 1.79e+02 | 8.67e-04 | 4.87e-01 | 6.1 | — | 3.98e-02 | — |
| N1 ReLU3 kinkfree S=4 | 3 | ok | 0 | 4.18e-11 | 4.24e+02 | 1.98e+02 | 1.32e-14 | 4.68e-01 | 0.122 | 40 / 0 / 0 | 6.75e-16 | — |
| N2 ReLU3 S=4 | 3 | ok | 0 | 4.73e-11 | 4.24e+02 | 1.98e+02 | 1.08e-14 | 4.68e-01 | 0.12 | 40 / 0 / 0 | 8.10e-16 | — |
| N2 tanh S=4 | 3 | unconverged | 20 | 1.24e-02 | 4.24e+02 | 1.98e+02 | 3.84e-05 | 4.68e-01 | 0.49 | — | 3.87e-05 | — |
| N2 tanh S=8 | 3 | unconverged | 20 | 1.02e-03 | 4.18e+02 | 1.93e+02 | 8.38e-05 | 4.73e-01 | 2.23 | — | 1.27e-02 | — |
| N1 ReLU3 kinkfree S=4 | 10 | ok | 0 | 5.46e-12 | 4.27e+02 | 1.99e+02 | 1.22e-14 | 4.66e-01 | 0.0413 | 12 / 0 / 0 | 6.73e-16 | — |
| N2 ReLU3 S=4 | 10 | ok | 0 | 7.28e-12 | 4.27e+02 | 1.99e+02 | 2.22e-14 | 4.66e-01 | 0.0405 | 12 / 0 / 0 | 1.21e-15 | — |
| N2 tanh S=4 | 10 | unconverged | 6 | 2.28e-02 | 4.27e+02 | 1.99e+02 | 5.37e-05 | 4.66e-01 | 0.145 | — | 3.28e-05 | — |
| N2 tanh S=8 | 10 | unconverged | 6 | 2.38e-03 | 4.26e+02 | 1.98e+02 | 3.50e-05 | 4.68e-01 | 0.444 | — | 3.72e-03 | — |

- N1 ReLU3 kinkfree S=4: 5 runs, 2 not converged (40 %), 0 better than the linear front by 3×
- N2 ReLU3 S=4: 5 runs, 3 not converged (60 %), 0 better than the linear front by 3×
- N2 tanh S=4: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N2 tanh S=8: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N1 with every step P₃-equivalent and converged: 3 runs, max diff to CGVI(P₃) 1.19e-15

## P3_e0.9

| network | h Ω | status | unconv. steps | max residual | q error | H | L | ϖ | wall time [s] | P₃-equiv / spline / degen | diff to CGVI(P₃) | linear front at that time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N1 ReLU3 kinkfree S=4 | 0.1 | ok | 0 | 9.66e-13 | 1.24e+00 | 1.32e+00 | 1.62e-12 | 1.84e+00 | 4.41 | 1256 / 0 / 0 | 1.01e+00 | 1.51e-06 |
| N2 ReLU3 S=4 | 0.1 | unconverged | 5 | 3.82e-05 | 1.16e+00 | 1.63e+00 | 1.17e-03 | 1.61e+00 | 4.24 | 1251 / 5 / 0 | 1.04e+00 | 1.51e-06 |
| N2 tanh S=4 | 0.1 | unconverged | 628 | 1.29e-02 | 6.02e+00 | 8.61e-01 | 2.34e-02 | 9.63e-02 | 17.6 | — | 8.39e-01 | 1.51e-06 |
| N2 tanh S=8 | 0.1 | unconverged | 574 | 1.79e-04 | 2.64e-01 | 1.25e-02 | 6.11e-05 | 3.16e-03 | 30.4 | — | 9.93e-01 | 1.51e-06 |
| N1 ReLU3 kinkfree S=4 | 0.3 | unconverged | 3 | 5.11e+00 | 9.59e+01 | 1.30e+01 | 1.75e-01 | 3.08e+00 | 1.42 | 416 / 2 / 0 | 3.74e-01 | 1.51e-06 |
| N2 ReLU3 S=4 | 0.3 | unconverged | 2 | 7.17e-01 | 6.08e+01 | 4.68e+00 | 4.67e-01 | 9.63e-01 | 1.44 | 417 / 1 / 0 | 8.91e-01 | 1.51e-06 |
| N2 tanh S=4 | 0.3 | unconverged | 209 | 2.35e-01 | 4.87e+01 | 3.33e+00 | 3.59e-01 | 5.64e-01 | 6.16 | — | 8.95e-01 | 1.51e-06 |
| N2 tanh S=8 | 0.3 | unconverged | 208 | 9.02e+01 | 1.96e+01 | 3.34e+00 | 1.98e-01 | 8.64e-01 | 12.3 | — | 1.12e+00 | 1.51e-06 |
| N1 ReLU3 kinkfree S=4 | 1 | unconverged | 1 | 2.41e+00 | 7.00e+01 | 9.02e+00 | 3.95e+00 | 2.59e+00 | 0.418 | 122 / 2 / 0 | 2.14e+00 | 1.51e-06 |
| N2 ReLU3 S=4 | 1 | unconverged | 1 | 3.58e-01 | 6.48e+01 | 5.59e+00 | 1.20e+00 | 2.59e+00 | 0.421 | 122 / 2 / 0 | 6.95e-01 | 1.51e-06 |
| N2 tanh S=4 | 1 | unconverged | 62 | 8.78e-02 | 1.12e+02 | 1.95e+01 | 3.29e-01 | 1.10e+00 | 1.58 | — | 1.69e+00 | 1.51e-06 |
| N2 tanh S=8 | 1 | unconverged | 62 | 1.32e-01 | 3.75e+01 | 2.32e+00 | 2.43e-01 | 5.14e-01 | 3.32 | — | 1.09e+00 | 1.51e-06 |
| N1 ReLU3 kinkfree S=4 | 3 | ok | 0 | 4.55e-13 | 1.11e+02 | 1.32e+01 | 1.87e-13 | 5.57e-01 | 0.138 | 40 / 0 / 0 | 2.84e-15 | 3.31e-01 |
| N2 ReLU3 S=4 | 3 | ok | 0 | 5.68e-13 | 1.11e+02 | 1.32e+01 | 3.94e-13 | 5.57e-01 | 0.133 | 40 / 0 / 0 | 7.30e-15 | 3.31e-01 |
| N2 tanh S=4 | 3 | unconverged | 20 | 4.26e-03 | 1.10e+02 | 1.31e+01 | 3.56e-02 | 5.76e-01 | 0.451 | — | 5.46e-03 | 1.51e-06 |
| N2 tanh S=8 | 3 | unconverged | 20 | 1.09e-01 | 9.47e+01 | 1.01e+01 | 6.32e-01 | 2.97e-01 | 1.26 | — | 1.46e-01 | 1.51e-06 |
| N1 ReLU3 kinkfree S=4 | 10 | unconverged | 1 | 2.36e-08 | 1.32e+02 | 1.81e+01 | 6.56e-12 | 5.05e-01 | 0.0479 | 12 / 0 / 0 | 5.51e-14 | — |
| N2 ReLU3 S=4 | 10 | unconverged | 1 | 2.11e-08 | 1.32e+02 | 1.81e+01 | 1.73e-11 | 5.05e-01 | 0.0466 | 12 / 0 / 0 | 2.07e-13 | — |
| N2 tanh S=4 | 10 | unconverged | 6 | 4.37e-03 | 1.32e+02 | 1.81e+01 | 2.97e-03 | 5.06e-01 | 0.136 | — | 2.32e-04 | 3.31e-01 |
| N2 tanh S=8 | 10 | unconverged | 6 | 2.29e-02 | 1.09e+02 | 1.26e+01 | 4.83e-03 | 5.67e-01 | 0.406 | — | 1.73e-01 | 1.51e-06 |

- N1 ReLU3 kinkfree S=4: 5 runs, 3 not converged (60 %), 0 better than the linear front by 3×
- N2 ReLU3 S=4: 5 runs, 4 not converged (80 %), 0 better than the linear front by 3×
- N2 tanh S=4: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N2 tanh S=8: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N1 with every step P₃-equivalent and converged: 2 runs, max diff to CGVI(P₃) 1.01e+00

## P4_E0.02

| network | h Ω | status | unconv. steps | max residual | q error | H | wall time [s] | P₃-equiv / spline / degen | diff to CGVI(P₃) | linear front at that time |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 ReLU3 kinkfree S=4 | 0.1 | ok | 0 | 2.46e-14 | 6.57e-10 | 1.75e-11 | 62.5 | 20000 / 0 / 0 | 2.01e-12 | 7.03e-14 |
| N2 ReLU3 S=4 | 0.1 | ok | 0 | 1.84e-14 | 6.57e-10 | 1.75e-11 | 62.4 | 20000 / 0 / 0 | 2.34e-12 | 7.03e-14 |
| N2 tanh S=4 | 0.1 | unconverged | 10000 | 3.88e-05 | 3.19e-04 | 2.21e-03 | 208 | — | 1.95e-02 | 7.03e-14 |
| N2 tanh S=8 | 0.1 | ok | 0 | 8.42e-11 | 3.80e-08 | 7.51e-08 | 191 | — | 1.85e-07 | 7.03e-14 |
| N1 ReLU3 kinkfree S=4 | 0.3 | ok | 0 | 7.11e-15 | 4.75e-07 | 1.30e-08 | 21.1 | 6666 / 0 / 0 | 1.76e-12 | 7.03e-14 |
| N2 ReLU3 S=4 | 0.3 | unconverged | 2 | 2.78e-07 | 4.75e-07 | 2.88e-06 | 21 | 6664 / 2 / 0 | 5.87e-04 | 7.03e-14 |
| N2 tanh S=4 | 0.3 | unconverged | 3333 | 3.98e-05 | 8.33e-03 | 6.25e-02 | 92.5 | — | 4.79e-01 | 7.03e-14 |
| N2 tanh S=8 | 0.3 | unconverged | 3222 | 1.81e-07 | 1.31e-06 | 1.64e-05 | 55.9 | — | 1.04e-04 | 7.03e-14 |
| N1 ReLU3 kinkfree S=4 | 1 | ok | 0 | 7.66e-15 | 6.37e-04 | 2.57e-05 | 7.08 | 2000 / 0 / 0 | 8.60e-14 | 7.03e-14 |
| N2 ReLU3 S=4 | 1 | unconverged | 20 | 4.22e-06 | 8.55e-04 | 1.75e-03 | 7.15 | 1980 / 20 / 0 | 1.18e-02 | 7.03e-14 |
| N2 tanh S=4 | 1 | unconverged | 1000 | 1.02e-04 | 1.45e-02 | 1.09e-02 | 23.2 | — | 1.49e-01 | 7.03e-14 |
| N2 tanh S=8 | 1 | unconverged | 996 | 1.16e-07 | 7.94e-07 | 3.86e-06 | 39.6 | — | 5.36e-03 | 7.03e-14 |
| N1 ReLU3 kinkfree S=4 | 3 | ok | 0 | 1.35e-13 | 2.13e-01 | 5.16e-02 | 2.52 | 666 / 0 / 0 | 4.10e-11 | 7.03e-14 |
| N2 ReLU3 S=4 | 3 | unconverged | 26 | 2.39e-03 | 2.06e-01 | 5.70e-02 | 2.8 | 639 / 27 / 0 | 1.31e-01 | 7.03e-14 |
| N2 tanh S=4 | 3 | unconverged | 333 | 5.88e-03 | 1.25e-01 | 2.00e+00 | 10 | — | 2.25e+00 | 7.03e-14 |
| N2 tanh S=8 | 3 | unconverged | 333 | 1.47e-04 | 1.86e-04 | 2.14e-03 | 31.2 | — | 1.51e+00 | 7.03e-14 |
| N1 ReLU3 kinkfree S=4 | 10 | unconverged | 15 | 1.67e-01 | 2.01e+00 | 1.00e+00 | 2.2 | 200 / 0 / 0 | — | 7.03e-14 |
| N2 ReLU3 S=4 | 10 | unconverged | 49 | 2.08e-01 | 2.17e+00 | 1.00e+00 | 1.71 | 164 / 36 / 0 | — | 7.03e-14 |
| N2 tanh S=4 | 10 | unconverged | 100 | 3.67e-01 | 1.99e+00 | 2.99e+00 | 2.35 | — | — | 7.03e-14 |
| N2 tanh S=8 | 10 | failed:NonlinearSolverException | 36 | 1.13e+09 | — | — | 1.66 | — | — | 7.03e-14 |

- N1 ReLU3 kinkfree S=4: 5 runs, 1 not converged (20 %), 0 better than the linear front by 3×
- N2 ReLU3 S=4: 5 runs, 4 not converged (80 %), 0 better than the linear front by 3×
- N2 tanh S=4: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N2 tanh S=8: 5 runs, 4 not converged (80 %), 0 better than the linear front by 3×
- N1 with every step P₃-equivalent and converged: 4 runs, max diff to CGVI(P₃) 4.10e-11

## P4_E0.135

| network | h Ω | status | unconv. steps | max residual | q error | H | wall time [s] | P₃-equiv / spline / degen | diff to CGVI(P₃) | linear front at that time |
|---|---|---|---|---|---|---|---|---|---|---|
| N1 ReLU3 kinkfree S=4 | 0.1 | ok | 0 | 6.48e-14 | 8.08e-08 | 5.51e-11 | 63.1 | 20000 / 0 / 0 | 1.73e+00 | 4.25e-12 |
| N2 ReLU3 S=4 | 0.1 | unconverged | 1 | 2.01e-06 | 8.08e-08 | 1.96e-06 | 63.2 | 19999 / 1 / 0 | 1.33e+00 | 4.25e-12 |
| N2 tanh S=4 | 0.1 | unconverged | 10000 | 7.35e-05 | 1.13e+00 | 1.27e-02 | 288 | — | 1.74e+00 | 4.25e-12 |
| N2 tanh S=8 | 0.1 | unconverged | 783 | 2.54e-10 | 3.99e-07 | 5.32e-08 | 220 | — | 1.75e+00 | 4.25e-12 |
| N1 ReLU3 kinkfree S=4 | 0.3 | ok | 0 | 2.04e-14 | 6.06e-05 | 4.17e-08 | 22.3 | 6666 / 0 / 0 | 1.64e+00 | 4.25e-12 |
| N2 ReLU3 S=4 | 0.3 | unconverged | 2 | 2.36e-07 | 6.06e-05 | 6.29e-06 | 22.4 | 6664 / 2 / 0 | 1.57e+00 | 4.25e-12 |
| N2 tanh S=4 | 0.3 | unconverged | 3333 | 7.50e-05 | 1.19e-01 | 1.33e-02 | 98.1 | — | 1.66e+00 | 4.25e-12 |
| N2 tanh S=8 | 0.3 | unconverged | 3302 | 1.18e-06 | 1.86e-04 | 1.18e-05 | 61.9 | — | 1.67e+00 | 4.25e-12 |
| N1 ReLU3 kinkfree S=4 | 1 | ok | 0 | 2.31e-14 | 1.52e-01 | 1.74e-04 | 7.43 | 2000 / 0 / 0 | 1.56e+00 | 4.25e-12 |
| N2 ReLU3 S=4 | 1 | unconverged | 26 | 8.54e-06 | 1.98e-01 | 6.41e-04 | 7.48 | 1974 / 26 / 0 | 1.63e+00 | 4.25e-12 |
| N2 tanh S=4 | 1 | unconverged | 1000 | 2.85e-04 | 7.74e-01 | 1.03e-02 | 22.6 | — | 1.53e+00 | 4.25e-12 |
| N2 tanh S=8 | 1 | unconverged | 1000 | 1.90e-05 | 4.80e-03 | 1.16e-04 | 47.5 | — | 1.58e+00 | 4.25e-12 |
| N1 ReLU3 kinkfree S=4 | 3 | unconverged | 18 | 3.49e-02 | 1.51e+00 | 2.66e-01 | 2.88 | 662 / 4 / 0 | 1.48e+00 | 2.16e-11 |
| N2 ReLU3 S=4 | 3 | unconverged | 46 | 4.93e-02 | 1.69e+00 | 1.65e-01 | 3.33 | 629 / 37 / 0 | 1.63e+00 | 4.25e-12 |
| N2 tanh S=4 | 3 | failed:NonlinearSolverException | 50 | 3.56e+08 | — | — | 1.37 | — | — | 2.16e-11 |
| N2 tanh S=8 | 3 | unconverged | 333 | 3.52e-04 | 8.78e-02 | 4.71e-04 | 12.5 | — | 1.57e+00 | 4.25e-12 |
| N1 ReLU3 kinkfree S=4 | 10 | unconverged | 14 | 3.22e-01 | 1.39e+00 | 1.00e+00 | 1.9 | 200 / 0 / 0 | — | 2.16e-11 |
| N2 ReLU3 S=4 | 10 | unconverged | 47 | 8.97e-01 | 1.15e+00 | 1.00e+00 | 2.24 | 169 / 31 / 0 | — | 2.16e-11 |
| N2 tanh S=4 | 10 | unconverged | 100 | 8.04e-01 | 1.33e+00 | 1.00e+00 | 2.54 | — | — | 2.16e-11 |
| N2 tanh S=8 | 10 | failed:NonlinearSolverException | 7 | 1.76e+08 | — | — | 0.407 | — | — | 2.75e-11 |

- N1 ReLU3 kinkfree S=4: 5 runs, 2 not converged (40 %), 0 better than the linear front by 3×
- N2 ReLU3 S=4: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N2 tanh S=4: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N2 tanh S=8: 5 runs, 5 not converged (100 %), 0 better than the linear front by 3×
- N1 with every step P₃-equivalent and converged: 3 runs, max diff to CGVI(P₃) 1.73e+00
