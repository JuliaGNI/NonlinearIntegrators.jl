#######
#General functions that are used in the network integrators
#######

function first_order_central_difference(f,x;ϵ=0.00001)
    return (f(x+ϵ)-f(x-ϵ))/(2*ϵ)
end

function first_order_forward_difference(f,x;ϵ=0.00001)
    return (f(x+ϵ)-f(x))/ϵ
end


function mse_loss(x,y::AbstractArray{T},NN,ps;λ=1000,μ = 0.00001) where T
    y_pred = NN(x,ps)
    mse_loss = mean(abs,y_pred - y) + λ*abs2(y_pred[1] - y[1])
    return mse_loss
end

function basis_first_order_central_difference(NN,ps,quad_nodes;ϵ=0.00001)
    bd = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([quad_nodes-ϵ],ps[1:end-1])
    fd = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([quad_nodes+ϵ],ps[1:end-1])
    return (fd .- bd) ./ (2*ϵ)
end