function first_order_central_difference(f,x;ϵ=0.00001)
    return (f(x+ϵ)-f(x-ϵ))/(2*ϵ)
end

function first_order_forward_difference(f,x;ϵ=0.00001)
    return (f(x+ϵ)-f(x))/ϵ
end


function mse_loss(x,y::AbstractArray{T},NN,ps;λ=0.0,μ = 0.00001) where T
    y_pred = NN(x,ps)
    loss = mean(abs,y_pred - y) + λ*abs2(y_pred[1] - y[1])
    return loss
end

function basis_first_order_central_difference(NN,ps,x;ϵ=0.00001)
    bd = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([x-ϵ],ps[1:end-1])
    fd = AbstractNeuralNetworks.Chain(NN.layers[1:end-1]...)([x+ϵ],ps[1:end-1])
    return (fd .- bd) ./ (2*ϵ)
end

function OneLayerbasis_first_order_central_difference(NN,ps,st,x;ϵ=0.00001)
    bd = NN[1]([x .- ϵ],ps[1],st[1])[1]
    fd = NN[1]([x .+ ϵ],ps[1],st[1])[1]
    return (fd .- bd) ./ (2*ϵ)
end

function vector_central_difference(basis,ps,st,x;ϵ=0.00001)
    local NN = basis.NN
    bd = NN[1](x .- ϵ,ps[1],st[1])[1]
    fd = NN[1](x .+ ϵ,ps[1],st[1])[1]
    return (fd .- bd) ./ (2*ϵ)
end

function vector_central_difference(basis,ps,x;ϵ=0.00001)
    local NN = basis.NN
    bd = (NN.layers[1])(x .- ϵ,ps[1])
    fd = (NN.layers[1])(x .+ ϵ,ps[1])
    return (fd .- bd) ./ (2*ϵ)
    
end

function basis_first_order_central_difference(NN,ps,st,x;ϵ=0.00001)
    bd = NN([x-ϵ],ps,st)[1]
    fd = NN([x+ϵ],ps,st)[1]
    return (fd .- bd) ./ (2*ϵ)
end

function vector_mse_loss(x,y,model, ps, st;λ=1000)
    y_pred, st = model(x, ps, st)
    loss = mean(abs2,y_pred - y) + λ*sum(abs2,y_pred[:,1]-y[:,1])
    return loss, ps,()
end

function vector_mse_energy_loss(x,y,model,ps,st,problem_module,params,initial_hamiltonian;λ=1000,ϵ = 0.00001,μ = 0.1)
    y_pred, st = model(x, ps, st)
    v_pred = (model(x .+ ϵ, ps, st)[1] - model(x .- ϵ, ps, st)[1])/(2*ϵ)
    hamiltonian_pred = [problem_module.ϑ(0.0, y_pred[:,i], v_pred[:,i], params)'*v_pred[:,i]- problem_module.lagrangian(0.0, y_pred[:,i], v_pred[:,i], params) for i in 1:size(y_pred,2)]
    energy_loss = mean(abs2,y_pred - y) + λ*sum(abs2,y_pred[:,1]-y[:,1]) + μ*sum(abs2,hamiltonian_pred .- initial_hamiltonian)
    return energy_loss, ps,()
end
