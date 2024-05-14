#######
#General functions that are used in the network integrators
#######

function first_order_central_difference(f,x;ϵ=0.00001)
    return (f(x+ϵ)-f(x-ϵ))/(2*ϵ)
end

function first_order_forward_difference(f,x;ϵ=0.00001)
    return (f(x+ϵ)-f(x))/ϵ
end

# @kernel function index_first_element_kernel(output::AbstractArray{T}, y::AbstractArray{T}) where T
#     i = @index(Global)
#     output[i] = y[i]
    
#     nothing
# end

# function get_first_element(y::AbstractVector{T}) where T
#     backend = KernelAbstractions.get_backend(y)
#     output = KernelAbstractions.zeros(T, 1)
#     index_first_element! = index_first_element!

#     index_first_element!(output, y, ndrange = length(output))

#     sum(output)
# end

function mse_loss(x,y,NN,ps;λ=1000)
    y_pred = NN(x,ps)
    index_vec = vcat(CUDA.ones(1), CUDA.zeros(10))
    mse_loss = mean(abs,y_pred - y) + λ*abs2(index_vec'*(y_pred - y))
    return mse_loss
end

function basis_first_order_central_difference(NN,ps,quad_nodes;ϵ=0.00001)
    bd = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([quad_nodes-ϵ],ps[1:end-1])
    fd = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([quad_nodes+ϵ],ps[1:end-1])
    return (fd .- bd) ./ (2*ϵ)
end