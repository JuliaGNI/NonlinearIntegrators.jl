#######
#General functions that are used in the network integrators
#######
using NonlinearIntegrators
using CompactBasisFunctions

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
    mse_loss = mean(abs2,y_pred - y) + λ*sum(abs2,y_pred[:,1]-y[:,1])
    return mse_loss, ps,()
end

function vector_mse_energy_loss(x,y,model,ps,st,problem_module,params,initial_hamiltonian;λ=1000,ϵ = 0.00001,μ = 0.1)
    y_pred, st = model(x, ps, st)
    v_pred = (model(x .+ ϵ, ps, st)[1] - model(x .- ϵ, ps, st)[1])/(2*ϵ)
    hamiltonian_pred = [problem_module.ϑ(0.0, y_pred[:,i], v_pred[:,i], params)'*v_pred[:,i]- problem_module.lagrangian(0.0, y_pred[:,i], v_pred[:,i], params) for i in 1:size(y_pred,2)]
    energy_loss = mean(abs2,y_pred - y) + λ*sum(abs2,y_pred[:,1]-y[:,1]) + μ*sum(abs2,hamiltonian_pred .- initial_hamiltonian)
    return energy_loss, ps,()
end

function simpson_quadrature(N::Int)
    if N % 2 != 0
        error("N must be even for Simpson's rule.")
    end
    
    # Step size
    h = 1.0 / N
        
    # Generate weights
    w = zeros(Float64, N + 1)
    for i in 1:(N + 1)
        if i == 1 || i == N + 1
            w[i] = h / 3 # First and last weights
        elseif i % 2 == 0
            w[i] = 4 * h / 3 # Even-indexed weights
        else
            w[i] = 2 * h / 3 # Odd-indexed weights
        end
    end
    
    return w
end

"""
    initial_trajectory!(sol, history, params, int, initial_trajectory)

Initial trajectory for the [`NonLinear_OneLayer_Lux`](@ref) integrator.
"""
function initial_trajectory!(sol, history, params, ::GeometricIntegrator, initial_trajectory::Extrapolation)
    error("For extrapolation $(initial_trajectory) method is not implemented!")
end

