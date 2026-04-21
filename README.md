# NonlinearIntegrators

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/NonlinearIntegrators.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaGNI.github.io/NonlinearIntegrators.jl/dev/)
[![Build Status](https://github.com/JuliaGNI/NonlinearIntegrators.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/NonlinearIntegrators.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/NonlinearIntegrators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/NonlinearIntegrators.jl)
<!-- 
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/N/NonlinearIntegrators.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/N/NonlinearIntegrators.html) -->


[NonlinearIntegrators.jl](https://github.com/JuliaGNI/NonlinearIntegrators.jl) is a affliated package in the [GeometricIntegrators.jl](https://github.com/JuliaGNI/GeometricIntegrators.jl) community. This package aims to generalize continuous Galerkin variational integrators from linear basis to nonlinear basis for achieving large time-step integration.

Till now, several options for nonlinear basis are available, but only neural network basis with one hidden layer is frequently used and well maintained.

- [x] One Hidden Layer Neural Network (Shallow Neural Network)
    - [x] Implemented with Lux
    - [x] Implemented with [GeometricMachineLearning.jl](https://github.com/JuliaGNI/GeometricMachineLearning.jl) and [SymbolicNeuralNetwork.jl](https://juliagni.github.io/SymbolicNeuralNetworks.jl)
- [x] Deep Neural Networks.
- [x] Nested Sindy


## installation
GeometricIntegrators.jl and all of its dependencies can be installed via the Julia REPL by typing
```
]add NonlinearIntegrators.jl
```

## Development

We are using git hooks, e.g., to enforce that all tests pass before pushing.
In order to activate these hooks, the following command must be executed once:
```
git config core.hooksPath .githooks
```
