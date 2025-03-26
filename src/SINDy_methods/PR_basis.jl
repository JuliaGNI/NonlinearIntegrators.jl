struct PR_Basis{T} <: Basis{T}
    q_expr
    W::Vector{Symbolics.Arr{Num,1}}
    dqdW

    v_expr
    dvdW

    problem_dim::Int
    W_sizes::Vector{Int}
    function PR_Basis{T}(q_expr::Vector{Num},W::Vector{Symbolics.Arr{Num, 1}},t::Num,D::Int64) where {T} 
        v_expr = [Symbolics.derivative(q_expr[i],t) for i in 1:D]

        dqdW_Mat = []
        dvdW_Mat = []
        W_sizes = map(length,W)

        for d in 1:D
            dqdW = zeros(Num,W_sizes[d])
            dvdW = zeros(Num,W_sizes[d])
            for i in 1:W_sizes[d]
                dqdW[i] = Symbolics.derivative(q_expr[d],W[d][i])
                dvdW[i] = Symbolics.derivative(v_expr[d],W[d][i])
            end
            dqdW = [Symbolics.eval(Symbolics.build_function(dqdW[i],W,t)) for i in 1:W_sizes[d]]
            dvdW = [Symbolics.eval(Symbolics.build_function(dvdW[i],W,t)) for i in 1:W_sizes[d]]

            push!(dqdW_Mat,dqdW)
            push!(dvdW_Mat,dvdW)
        end

        q_expr = [Symbolics.eval(Symbolics.build_function(q_expr[d],W[d],t)) for d in 1:D]
        v_expr = [Symbolics.eval(Symbolics.build_function(v_expr[d],W[d],t)) for d in 1:D]
        # dqdW = [Symbolics.eval(Symbolics.build_function(dqdW[i],W,t)) for i in 1:NDOFs]
        # dvdW = [Symbolics.eval(Symbolics.build_function(dvdW[i],W,t)) for i in 1:NDOFs]

        new{T}(q_expr,W,dqdW_Mat,v_expr,dvdW_Mat,D,W_sizes)
    end
end

# function Base.show(io::IO,basis::PR_Basis)
#     print(io, "\n")
#     print(io, "  =====================================", "\n")
#     print(io, "  ======PR Basis by Symbolics======", "\n")
#     print(io, "  =====================================", "\n")
#     print(io, "\n")
# end