"""
    VISEBasis{T}(q_expr, W, t, D)

Basis for [`VISE`](@ref): a symbolic ansatz `q_expr[d](W[d], t)` per degree of freedom `d`,
together with its time derivative and its derivatives with respect to every weight.

`q_expr` is a vector of `D` Symbolics expressions, `W` a vector of `D` symbolic weight arrays,
and `t` the symbolic time variable. Everything is compiled at construction.

# Implementation

The compiled callables come from `Symbolics.build_function(…; expression = Val(false))`, which
returns a `RuntimeGeneratedFunction`. The previous form was
`Symbolics.eval(Symbolics.build_function(…))`, which had two costs:

  * **World age.** `eval` adds methods to the table in a *newer* world than the one the calling
    code is running in, so building a basis and evaluating it within the same function body
    raised `MethodError: … The applicable method may be too new`. It happened to work when the
    basis was built at top level, because top-level statements advance the world age between
    them — which is why the only test of this integrator built it that way, and why a test that
    used a helper function to build it failed.
  * **Typing.** `eval` returns `Any`, so `q_expr`, `dqdW`, `v_expr` and `dvdW` could not be
    given concrete field types no matter how the struct was declared, and every call through
    them in `components!` — `R × W_size × D` of them per residual evaluation — was a dynamic
    dispatch. `RuntimeGeneratedFunction` is a concrete type, so the fields below are
    parametrised on it and the calls resolve statically.
"""
struct VISEBasis{T,QE,DQ,VE,DV} <: Basis{T}
    q_expr::QE
    W::Vector{Symbolics.Arr{Num,1}}
    dqdW::DQ

    v_expr::VE
    dvdW::DV

    problem_dim::Int
    W_sizes::Vector{Int}

    function VISEBasis{T}(q_expr::Vector{Num}, W::Vector{Symbolics.Arr{Num,1}}, t::Num,
                          D::Int) where {T}
        v_expr = [Symbolics.derivative(q_expr[i], t) for i in 1:D]
        W_sizes = map(length, W)

        # `expression = Val(false)` → a compiled `RuntimeGeneratedFunction`, not an `Expr` that
        # has to be `eval`'d. `map` over a range gives a concretely typed `Vector`, where the
        # previous `mat = []` / `push!` built a `Vector{Any}` whose elements were `Any` too.
        compile(expr, d) = Symbolics.build_function(expr, W[d], t; expression = Val(false))

        dqdW_Mat = [[compile(Symbolics.derivative(q_expr[d], W[d][i]), d) for i in 1:W_sizes[d]]
                    for d in 1:D]
        dvdW_Mat = [[compile(Symbolics.derivative(v_expr[d], W[d][i]), d) for i in 1:W_sizes[d]]
                    for d in 1:D]

        q_funcs = [compile(q_expr[d], d) for d in 1:D]
        v_funcs = [compile(v_expr[d], d) for d in 1:D]

        new{T,typeof(q_funcs),typeof(dqdW_Mat),typeof(v_funcs),typeof(dvdW_Mat)}(
            q_funcs, W, dqdW_Mat, v_funcs, dvdW_Mat, D, W_sizes)
    end
end

function Base.show(io::IO, basis::VISEBasis{T}) where {T}
    print(io, "\n")
    print(io, "  =====================================", "\n")
    print(io, "  ========VISE Symbolic Basis==========", "\n")
    print(io, "  =====================================", "\n")
    print(io, "\n")
    print(io, "    Element type        = ", T, "\n")
    print(io, "    Problem dimension D = ", basis.problem_dim, "\n")
    print(io, "    Weights per DOF     = ", basis.W_sizes, "\n")
    print(io, "\n")
end
