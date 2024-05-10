using CompactBasisFunctions
using ContinuumArrays

struct OneLayerNetwork{T,BT}<:OneLayerNetBasis{T} 
    b::BT
    activation
    S::Int
    
    function OneLayerNetwork{T}(activation,S,W,bias) where {T}
        b = collect(y -> activation(W[i]*y + bias[i]) for i in 1:S)
        new{T,typeof(b)}(b,activation,S)
    end
    OneLayerNetwork(activation,S,W::AbstractVecOrMat{T},bias::AbstractVecOrMat{T}) where {T} = OneLayerNetwork{T}(activation,S,W,bias)
end

function Base.show(io::IO,basis::OneLayerNetwork)
    print(io, "\n")
    print(io, "  ===============================", "\n")
    print(io, "  ====One Layer Network Basis====", "\n")
    print(io, "  ===============================", "\n")
    print(io, "\n")
    print(io, "    Activation function σ  = ", basis.activation, "\n")
    print(io, "    Last Layer Nodes, Number of Basis S  = ", basis.S, "\n")
    print(io, "    Trainable NN Parameters Amount  = ",3*basis.S, "\n")
    print(io, "\n")
end

(L::OneLayerNetwork)(x::Number, j::Integer) = L.b[j](x)
Base.eachindex(L::OneLayerNetwork) = Base.eachindex(L.b)
Base.axes(L::OneLayerNetwork) = (Inclusion(0..1), eachindex(L))

Base.getindex(L::OneLayerNetwork, j::Integer)  = L.b[j]
Base.getindex(L::OneLayerNetwork, x::Number, j::Integer) = L(x,j)
Base.getindex(L::OneLayerNetwork, x::Number,  ::Colon) = [b(x) for b in L.b]
Base.getindex(L::OneLayerNetwork, X::AbstractVector, j::Integer) = L.(X,j)
Base.getindex(L::OneLayerNetwork, X::AbstractVector,  ::Colon) = [b(x) for x in X, b in L.b]

CompactBasisFunctions.basis(L::OneLayerNetwork) = L.b
CompactBasisFunctions.nbasis(L::OneLayerNetwork) = L.S

ContinuumArrays.grid(L::OneLayerNetwork) = L.x