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

function draw_comparison(titlename,ode_problem,problem_hamiltonian,truth_name,truth,names,sols...;problem_name = "Problem",h=1,plotrange = 50,save_path ="" )
    """
        ode_problem: the ode problem
        names: Array{String}, name of the solution for plot
    """
    p = plot(layout=@layout([a b;c]),label ="",size = (700,700),plot_title = titlename)# d;e

    plot!(p[1],0:0.1:plotrange,collect(truth.q[:,1]),label = truth_name,ylims = (-2,2))
    for prefs in zip(names,sols)
        plot!(p[1],0:h:plotrange,collect(prefs[2].q[:,1]),label=prefs[1],xaxis="time",yaxis="q₁")
        # scatter!(p[1],0:h:plotrange,collect(prefs[2].q[:,1]),label="",markersize = 1,ylims = (-3,3))
    end

    # plot!(p[2],0:0.1:plotrange,collect(truth.q[:,2]),label = truth_name,ylims = (-2,2))
    # for prefs in zip(names,sols)
    #     plot!(p[2],0:h:plotrange,collect(prefs[2].q[:,2]),label= prefs[1],xaxis="time",yaxis="q₂")
    #     # scatter!(p[2],0:h:plotrange,collect(prefs[2].q[:,2]),label="",markersize = 1,ylims = (-3,3))
    # end

    plot!(p[2],0:0.1:plotrange,collect(truth.p[:,1]),label = truth_name,ylims = (-5,5))
    for prefs in zip(names,sols)
        plot!(p[2],0:h:plotrange,collect(prefs[2].p[:,1]),label=prefs[1],xaxis="time",yaxis="p₁")
        # scatter!(p[3],0:h:plotrange,collect(prefs[2].p[:,1]),label="",markersize = 1,ylims = (-60,60))
    end

    # plot!(p[4],0:0.1:plotrange,collect(truth.p[:,2]),label = truth_name,ylims = (-5,5))
    # for prefs in zip(names,sols)
    #     plot!(p[4],0:h:plotrange,collect(prefs[2].p[:,2]),label=prefs[1],xaxis="time",yaxis="p₂")
    #     # scatter!(p[4],0:h:plotrange,collect(prefs[2].p[:,2]),label="",markersize = 1,ylims = (-60,60))
    # end

    true_ham = [problem_hamiltonian(0,q,p,ode_problem.parameters) for (q,p) in zip(collect(truth.q[:]),collect(truth.p[:]))]
    # plot!(p[3],0:0.1:plotrange,true_ham,label = truth_name,)
    for prefs in zip(names,sols)
        ham = [problem_hamiltonian(0,q,p,ode_problem.parameters) for (q,p) in zip(collect(prefs[2].q[:]),collect(prefs[2].p[:]))]
        plot!(p[3],0:h:plotrange,(ham .- true_ham[1])/true_ham[1],label= prefs[1],xaxis="time",yaxis="Relative Hamiltonian error")
        # scatter!(p[5],0:h:plotrange,ham,label="",markersize = 1,ylims = (-60,-40))
    end
    
    
    if save_path != ""
        savefig(save_path)
    end
    return p
end