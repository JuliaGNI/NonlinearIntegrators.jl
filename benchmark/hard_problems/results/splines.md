# Free-knot vs fixed-knot splines vs Lagrange polynomials (degree 3, Gauss(4) per cell)

Entries: q error / wall time in s. `fb` = fallback steps of the knot equation. Verdict: FreeSpline is **better** iff converged, fallbacks ≤ 5% of the steps, and its q error < 1/3 of every Lagrange or FixedSpline run with wall time ≤ its own (at least one such run).

## FrequencyModulatedOscillator_eps0.01

| h Ω | P3 | P4 | P5 | P6 | P3 x2 | P3 x3 | P3 x4 | S3 n=1 | S3 n=2 | S3 n=3 | S3 free n=1 | S3 free n=2 | S3 free n=3 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 1.4e-08 / 0.29 | 1.1e-12 / 0.25 | 7.1e-13 / 0.27 | 4.5e-13 / 0.3 | 2.2e-10 / 0.51 | 1.9e-11 / 0.87 | 3.6e-12 / 1.1 | 3.8e-10 / 0.23 | 4.8e-11 / 0.27 | 9.1e-12 / 0.34 | 3.8e-10 / 5.4 (fb 0) | 4.8e-11 / 5.4 (fb 0) | 9.1e-12 / 6.1 (fb 0) | – |
| 0.3 | 1.0e-05 / 0.15 | 7.2e-09 / 0.081 | 3.4e-12 / 0.14 | 2.4e-13 / 0.12 | 1.6e-07 / 0.22 | 1.4e-08 / 0.23 | 2.5e-09 / 0.38 | 2.8e-07 / 0.07 | 3.5e-08 / 0.081 | 6.6e-09 / 0.1 | 2.8e-07 / 2 (fb 10) | 1.4e-07 / 4.3 (fb 415) | failed:SingularException NaN / 1.8 (fb 100) | – |
| 1.0 | 1.3e-02 / 0.02 | 1.0e-04 / 0.024 | 5.4e-07 / 0.028 | 2.0e-09 / 0.032 | 2.1e-04 / 0.11 | 1.9e-05 / 0.14 | 3.4e-06 / 0.16 | 4.4e-04 / 0.018 | 5.3e-05 / 0.021 | 9.6e-06 / 0.025 | 3.4e-04 / 0.65 (fb 3) | 3.2e-04 / 3.3 (fb 560) | failed:SingularException NaN / 0.79 (fb 71) | – |
| 3.0 | 1.1e+00 / 0.0076 | 4.2e-01 / 0.0089 | 2.2e-02 / 0.01 | 7.6e-04 / 0.012 | 1.3e-01 / 0.014 | 1.3e-02 / 0.02 | 2.4e-03 / 0.11 | 4.7e-01 / 0.0079 | 6.4e-02 / 0.0092 | 1.0e-02 / 0.01 | 1.6e+00 / 0.26 (fb 21) | failed:SingularException NaN / 0.45 (fb 80) | failed:SingularException NaN / 0.77 (fb 72) | – |
| 10.0 | unconverged 2.3e+07 / 0.0043 | 1.7e+03 / 0.0049 | 3.6e+02 / 0.0049 | 3.0e+01 / 0.0053 | unconverged 3.4e+05 / 0.0066 | 1.6e+00 / 0.007 | 1.7e+00 / 0.0089 | 1.4e+04 / 0.0051 | unconverged 8.1e+08 / 0.0054 | 7.4e+00 / 0.0058 | unconverged 3.2e+12 / 0.084 (fb 49) | unconverged 5.5e+05 / 0.22 (fb 61) | unconverged 1.1e+06 / 0.44 (fb 57) | – |

## HenonHeiles_E0.02

| h Ω | P3 | P4 | P5 | P6 | P3 x2 | P3 x3 | P3 x4 | S3 n=1 | S3 n=2 | S3 n=3 | S3 free n=1 | S3 free n=2 | S3 free n=3 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 6.6e-10 / 0.55 | 2.2e-13 / 0.57 | 2.0e-13 / 0.69 | 5.2e-13 / 0.79 | 1.0e-11 / 0.91 | 8.2e-13 / 1.2 | 1.1e-13 / 1.7 | 1.8e-11 / 0.48 | 2.2e-12 / 0.58 | 3.4e-13 / 0.7 | 1.8e-11 / 8.3 (fb 0) | 2.1e-12 / 8.5 (fb 0) | 3.0e-13 / 9.1 (fb 0) | – |
| 0.3 | 4.8e-07 / 0.21 | 3.7e-10 / 0.17 | 2.9e-13 / 0.2 | 1.3e-13 / 0.24 | 7.5e-09 / 0.37 | 6.6e-10 / 0.46 | 1.2e-10 / 0.74 | 1.3e-08 / 0.16 | 1.7e-09 / 0.2 | 3.1e-10 / 0.24 | 1.3e-08 / 3.1 (fb 0) | 1.4e-09 / 2.9 (fb 0) | 3.1e-10 / 3.1 (fb 0) | – |
| 1.0 | 6.4e-04 / 0.12 | 5.4e-06 / 0.057 | 4.9e-08 / 0.066 | 3.3e-10 / 0.076 | 1.0e-05 / 0.15 | 9.2e-07 / 0.19 | 1.6e-07 / 0.24 | 2.2e-05 / 0.064 | 2.6e-06 / 0.079 | 4.7e-07 / 0.096 | 2.1e-05 / 1.2 (fb 1) | 2.1e-06 / 6.2 (fb 730) | 4.1e-07 / 3.5 (fb 95) | – |
| 3.0 | 2.1e-01 / 0.022 | 1.8e-02 / 0.026 | 1.0e-03 / 0.029 | 8.7e-05 / 0.033 | 5.7e-03 / 0.11 | 5.1e-04 / 0.047 | 9.3e-05 / 0.14 | 2.1e-02 / 0.03 | 2.5e-03 / 0.037 | 4.4e-04 / 0.044 | 2.0e-02 / 0.53 (fb 4) | 1.8e-03 / 2.7 (fb 265) | 2.7e-04 / 5 (fb 261) | – |
| 10.0 | failed:NonlinearSolverException NaN / 0.13 | failed:NonlinearSolverException NaN / 0.068 | failed:NonlinearSolverException NaN / 0.077 | failed:NonlinearSolverException NaN / 0.088 | failed:NonlinearSolverException NaN / 0.13 | 3.3e-01 / 0.021 | 6.2e-02 / 0.024 | failed:NonlinearSolverException NaN / 0.06 | failed:NonlinearSolverException NaN / 0.057 | failed:NonlinearSolverException NaN / 0.062 | unconverged 1.6e+00 / 0.48 (fb 28) | failed:SingularException NaN / 0.15 (fb 5) | unconverged 6.7e-01 / 0.93 (fb 31) | – |

## KeplerProblem_e0.9

| h Ω | P3 | P4 | P5 | P6 | P3 x2 | P3 x3 | P3 x4 | S3 n=1 | S3 n=2 | S3 n=3 | S3 free n=1 | S3 free n=2 | S3 free n=3 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 2.9e-04 / 0.41 | 1.3e-06 / 0.39 | 6.8e-09 / 0.45 | 5.8e-11 / 0.53 | 4.6e-06 / 0.74 | 4.0e-07 / 1.1 | 7.2e-08 / 1.4 | 8.9e-06 / 0.38 | 1.1e-06 / 0.45 | 2.0e-07 / 0.54 | 9.0e-06 / 5.5 (fb 2) | 9.0e-07 / 5.8 (fb 11) | 1.5e-07 / 6.4 (fb 16) | – |
| 0.03 | 1.7e-01 / 0.18 | 8.1e-03 / 0.12 | 1.6e-04 / 0.14 | 2.8e-05 / 0.17 | 3.1e-03 / 0.29 | 2.7e-04 / 0.33 | 4.8e-05 / 0.52 | 1.1e-02 / 0.14 | 1.4e-03 / 0.17 | 2.2e-04 / 0.22 | 1.1e-02 / 1.9 (fb 3) | 8.9e-04 / 2.1 (fb 11) | 2.5e-04 / 2.8 (fb 33) | – |
| 0.1 | 1.1e+00 / 0.11 | 1.4e+00 / 0.039 | 1.1e+00 / 0.043 | 6.8e-01 / 0.053 | 6.0e-01 / 0.13 | 2.2e-01 / 0.17 | 5.8e-02 / 0.2 | 1.7e+03 / 0.042 | unconverged 5.8e+03 / 0.051 | 2.2e-01 / 0.063 | 1.0e+00 / 0.7 (fb 6) | 9.7e-01 / 1.3 (fb 57) | 7.3e-01 / 1.8 (fb 41) | – |
| 0.3 | 1.3e+02 / 0.012 | 1.5e+02 / 0.013 | 1.1e+02 / 0.015 | unconverged 1.9e+02 / 0.019 | unconverged 8.4e+01 / 0.023 | 2.2e+00 / 0.034 | 1.0e+00 / 0.12 | 1.5e+01 / 0.012 | unconverged 9.6e+01 / 0.018 | unconverged 7.0e+01 / 0.022 | unconverged 9.4e+01 / 0.2 (fb 0) | unconverged 5.8e+01 / 0.27 (fb 13) | unconverged 1.1e+00 / 1.7 (fb 71) | – |
| 1.0 | unconverged 6.2e+01 / 0.0047 | 1.5e+02 / 0.0051 | unconverged 7.4e+01 / 0.0059 | unconverged 5.0e+01 / 0.0071 | unconverged 1.4e+02 / 0.01 | 1.9e+02 / 0.011 | 1.6e+02 / 0.014 | unconverged 1.0e+02 / 0.0053 | unconverged 7.2e+01 / 0.0063 | unconverged 6.5e+01 / 0.008 | 7.0e+02 / 0.062 (fb 2) | unconverged 2.7e+03 / 0.074 (fb 0) | unconverged 6.3e+01 / 0.12 (fb 2) | – |
| 3.0 | 1.1e+02 / 0.0018 | 9.4e+01 / 0.002 | 7.0e+01 / 0.0022 | 2.5e+01 / 0.0023 | 6.1e+01 / 0.0028 | unconverged 6.0e+01 / 0.0047 | 5.2e+01 / 0.005 | unconverged 4.3e+01 / 0.0034 | unconverged 8.8e+01 / 0.004 | unconverged 7.9e+01 / 0.0047 | unconverged 4.2e+01 / 0.03 (fb 1) | unconverged 5.5e+01 / 0.032 (fb 1) | unconverged 5.9e+01 / 0.044 (fb 2) | – |
| 10.0 | 1.3e+02 / 0.0019 | 1.3e+02 / 0.002 | 1.2e+02 / 0.0021 | 1.2e+02 / 0.0021 | 1.2e+02 / 0.0017 | 1.1e+02 / 0.0018 | 1.0e+02 / 0.0022 | 1.2e+02 / 0.0021 | 1.1e+02 / 0.0021 | 1.0e+02 / 0.0023 | 1.3e+02 / 0.0097 (fb 1) | failed:SingularException NaN / 0.0077 (fb 1) | failed:SingularException NaN / 0.011 (fb 1) | – |

