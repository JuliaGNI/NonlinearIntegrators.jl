# Linear baselines (hard problems, phase 1)

L1 = CGVI(P_s, R = s + 1), s = 2..6, with m = 1..4 substeps of h/m; L2 = Gauss(s), s = 1..6,
without substeps. h Ω ∈ {0.1, 0.3, 1, 3, 10}, Ω = 1 (ω ≈ 1 for the frequency-modulated
oscillator and Hénon–Heiles, the mean motion for the Kepler problem). q error: max over the
grid n h (t ≤ Terr) of the relative ∞-norm error. A run is
converged if no solve threw and the residual ∞-norm of every step is at most 1e-10. Wall
times include no compilation; runs under 0.1 s are the minimum of 3 repeats.

## FrequencyModulatedOscillator_eps0.001

Reference self-check (dt vs dt/2): 2.46e-13. Errors below 10× this value (†) are not resolved by the reference.

- CGVI: 100 runs, 8 not converged (8.0 %), 16 converged with q error ≥ 1
- Gauss: 30 runs, 0 not converged (0.0 %), 14 converged with q error ≥ 1

Pareto front of all linear methods (converged runs with q error < 1.0):

| method | h Ω | q error | wall time [s] | Newton iterations |
|---|---|---|---|---|
| Gauss(5) | 3 | 4.20e-01 | 0.0966 | 2886 |
| Gauss(6) | 3 | 1.48e-02 | 0.109 | 2988 |
| CGVI(P6, m=1) | 3 | 7.60e-03 | 0.213 | 5694 |
| Gauss(4) | 1 | 2.06e-03 | 0.239 | 6283 |
| CGVI(P6, m=4) | 10 | 9.51e-04 | 0.255 | 6867 |
| Gauss(5) | 1 | 1.07e-05 | 0.262 | 6283 |
| Gauss(6) | 1 | 3.91e-08 | 0.288 | 6283 |
| CGVI(P6, m=1) | 1 | 1.96e-08 | 0.584 | 14542 |
| CGVI(P6, m=3) | 3 | 1.96e-08 | 0.591 | 14542 |
| CGVI(P6, m=4) | 3 | 6.32e-10 | 0.762 | 18442 |
| Gauss(5) | 0.3 | 6.67e-11 | 0.868 | 20943 |
| Gauss(6) | 0.3 | 4.38e-13 † | 0.951 | 20943 |
| Gauss(5) | 0.1 | 1.75e-13 † | 2.75 | 62831 |

## FrequencyModulatedOscillator_eps0.01

Reference self-check (dt vs dt/2): 1.67e-13. Errors below 10× this value (†) are not resolved by the reference.

- CGVI: 100 runs, 3 not converged (3.0 %), 13 converged with q error ≥ 1
- Gauss: 30 runs, 0 not converged (0.0 %), 12 converged with q error ≥ 1

Pareto front of all linear methods (converged runs with q error < 1.0):

| method | h Ω | q error | wall time [s] | Newton iterations |
|---|---|---|---|---|
| Gauss(4) | 3 | 7.50e-01 | 0.01 | 270 |
| Gauss(5) | 3 | 4.23e-02 | 0.0112 | 287 |
| Gauss(6) | 3 | 1.49e-03 | 0.012 | 289 |
| CGVI(P6, m=1) | 3 | 7.65e-04 | 0.0225 | 573 |
| Gauss(4) | 1 | 2.06e-04 | 0.024 | 628 |
| CGVI(P6, m=4) | 10 | 1.01e-04 | 0.0265 | 677 |
| Gauss(5) | 1 | 1.07e-06 | 0.027 | 628 |
| Gauss(6) | 1 | 3.91e-09 | 0.0292 | 628 |
| CGVI(P6, m=3) | 3 | 1.96e-09 | 0.0595 | 1458 |
| CGVI(P6, m=1) | 1 | 1.96e-09 | 0.0597 | 1458 |
| CGVI(P6, m=4) | 3 | 6.34e-11 | 0.076 | 1850 |
| Gauss(5) | 0.3 | 6.77e-12 | 0.0866 | 2094 |
| Gauss(6) | 0.3 | 1.28e-13 † | 0.0949 | 2094 |
| Gauss(5) | 0.1 | 1.01e-13 † | 0.261 | 6283 |

## KeplerProblem_e0.99

- CGVI: 100 runs, 13 not converged (13.0 %), 87 converged with q error ≥ 1
- Gauss: 30 runs, 0 not converged (0.0 %), 30 converged with q error ≥ 1
- cross-check: 1 converged runs with an error of L above 1e-10

Pareto front of all linear methods (converged runs with q error < 1.0):

| method | h Ω | q error | wall time [s] | Newton iterations |
|---|---|---|---|---|

## KeplerProblem_e0.9

- CGVI: 100 runs, 31 not converged (31.0 %), 52 converged with q error ≥ 1
- Gauss: 30 runs, 9 not converged (30.0 %), 21 converged with q error ≥ 1
- cross-check: 1 converged runs with an error of L above 1e-10

Pareto front of all linear methods (converged runs with q error < 1.0):

| method | h Ω | q error | wall time [s] | Newton iterations |
|---|---|---|---|---|
| CGVI(P6, m=3) | 0.3 | 6.59e-01 | 0.104 | 2066 |
| CGVI(P6, m=1) | 0.1 | 6.35e-01 | 0.116 | 2056 |
| CGVI(P6, m=4) | 0.3 | 3.45e-01 | 0.127 | 2479 |
| CGVI(P4, m=2) | 0.1 | 3.31e-01 | 0.132 | 3339 |
| CGVI(P5, m=2) | 0.1 | 5.61e-02 | 0.161 | 3558 |
| CGVI(P6, m=2) | 0.1 | 2.22e-03 | 0.19 | 3736 |
| CGVI(P5, m=3) | 0.1 | 3.52e-04 | 0.255 | 5669 |
| CGVI(P6, m=3) | 0.1 | 6.66e-05 | 0.294 | 5814 |
| CGVI(P5, m=4) | 0.1 | 5.21e-05 | 0.35 | 7815 |
| CGVI(P6, m=4) | 0.1 | 1.51e-06 | 0.405 | 7945 |

## HenonHeiles_E0.02

Reference self-check (dt vs dt/2): 2.23e-13. Errors below 10× this value (†) are not resolved by the reference.

- CGVI: 100 runs, 11 not converged (11.0 %), 2 converged with q error ≥ 1
- Gauss: 30 runs, 6 not converged (20.0 %), 3 converged with q error ≥ 1

Pareto front of all linear methods (converged runs with q error < 1.0):

| method | h Ω | q error | wall time [s] | Newton iterations |
|---|---|---|---|---|
| Gauss(3) | 3 | 4.73e-01 | 0.0241 | 1523 |
| Gauss(4) | 3 | 2.72e-02 | 0.0266 | 1396 |
| Gauss(5) | 3 | 2.62e-03 | 0.031 | 1334 |
| Gauss(6) | 3 | 1.30e-04 | 0.0373 | 1315 |
| Gauss(4) | 1 | 1.14e-05 | 0.0594 | 2734 |
| CGVI(P6, m=4) | 10 | 1.01e-05 | 0.0706 | 1592 |
| Gauss(5) | 1 | 9.03e-08 | 0.0716 | 2723 |
| Gauss(6) | 1 | 7.24e-10 | 0.0846 | 2705 |
| CGVI(P6, m=1) | 1 | 3.27e-10 | 0.145 | 3107 |
| CGVI(P6, m=3) | 3 | 2.63e-10 | 0.146 | 3107 |
| CGVI(P6, m=4) | 3 | 8.80e-12 | 0.188 | 3955 |
| Gauss(5) | 0.3 | 5.10e-13 † | 0.204 | 6666 |
| Gauss(6) | 0.3 | 9.38e-14 † | 0.238 | 6666 |
| CGVI(P6, m=2) | 1 | 8.47e-14 † | 0.252 | 5014 |
| Gauss(4) | 0.1 | 7.98e-14 † | 0.404 | 10000 |
| CGVI(P5, m=4) | 1 | 7.03e-14 † | 0.477 | 10951 |

## HenonHeiles_E0.135

Reference self-check (dt vs dt/2): 1.23e-10. Errors below 10× this value (†) are not resolved by the reference.

- CGVI: 100 runs, 15 not converged (15.0 %), 9 converged with q error ≥ 1
- Gauss: 30 runs, 9 not converged (30.0 %), 4 converged with q error ≥ 1

Pareto front of all linear methods (converged runs with q error < 1.0):

| method | h Ω | q error | wall time [s] | Newton iterations |
|---|---|---|---|---|
| Gauss(5) | 3 | 6.44e-01 | 0.0376 | 1783 |
| Gauss(6) | 3 | 4.71e-03 | 0.0453 | 1749 |
| Gauss(4) | 1 | 4.50e-03 | 0.0632 | 2992 |
| Gauss(5) | 1 | 1.21e-04 | 0.0747 | 2995 |
| Gauss(6) | 1 | 1.69e-06 | 0.0898 | 2991 |
| CGVI(P6, m=1) | 1 | 9.65e-07 | 0.169 | 3797 |
| Gauss(4) | 0.3 | 6.37e-08 | 0.172 | 6666 |
| Gauss(5) | 0.3 | 2.17e-10 † | 0.202 | 6666 |
| Gauss(6) | 0.3 | 5.01e-11 † | 0.239 | 6666 |
| CGVI(P6, m=2) | 1 | 2.75e-11 † | 0.319 | 6870 |
| CGVI(P4, m=1) | 0.1 | 2.16e-11 † | 1.12 | 30686 |
| CGVI(P5, m=3) | 0.1 | 4.25e-12 † | 3.15 | 65871 |
