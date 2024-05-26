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

function draw_comparison(titlename,ode_problem,problem_hamiltonian,truth_name,truth,names,sols...;problem_name = "Problem",h=1,plotrange = 50)
    """
        ode_problem: the ode problem
        names: Array{String}, name of the solution for plot
    """
    p = plot(layout=@layout([a b;c d;e]),label ="",size = (700,700),plot_title = titlename)

    plot!(p[1],0:0.1:plotrange,collect(truth.q[:,1]),label = truth_name,ylims = (-2,2))
    for prefs in zip(names,sols)
        plot!(p[1],0:h:plotrange,collect(prefs[2].q[:,1]),label=prefs[1],xaxis="time",yaxis="q₁")
        # scatter!(p[1],0:h:plotrange,collect(prefs[2].q[:,1]),label="",markersize = 1,ylims = (-3,3))
    end

    plot!(p[2],0:0.1:plotrange,collect(truth.q[:,2]),label = truth_name,ylims = (-2,2))
    for prefs in zip(names,sols)
        plot!(p[2],0:h:plotrange,collect(prefs[2].q[:,2]),label= prefs[1],xaxis="time",yaxis="q₂")
        # scatter!(p[2],0:h:plotrange,collect(prefs[2].q[:,2]),label="",markersize = 1,ylims = (-3,3))
    end

    plot!(p[3],0:0.1:plotrange,collect(truth.p[:,1]),label = truth_name,ylims = (-5,5))
    for prefs in zip(names,sols)
        plot!(p[3],0:h:plotrange,collect(prefs[2].p[:,1]),label=prefs[1],xaxis="time",yaxis="p₁")
        # scatter!(p[3],0:h:plotrange,collect(prefs[2].p[:,1]),label="",markersize = 1,ylims = (-60,60))
    end

    plot!(p[4],0:0.1:plotrange,collect(truth.p[:,2]),label = truth_name,ylims = (-5,5))
    for prefs in zip(names,sols)
        plot!(p[4],0:h:plotrange,collect(prefs[2].p[:,2]),label=prefs[1],xaxis="time",yaxis="p₂")
        # scatter!(p[4],0:h:plotrange,collect(prefs[2].p[:,2]),label="",markersize = 1,ylims = (-60,60))
    end

    true_ham = [problem_hamiltonian(0,q,p,ode_problem.parameters) for (q,p) in zip(collect(truth.q[:]),collect(truth.p[:]))]
    plot!(p[5],0:0.1:plotrange,true_ham,label = truth_name,)
    for prefs in zip(names,sols)
        ham = [problem_hamiltonian(0,q,p,ode_problem.parameters) for (q,p) in zip(collect(prefs[2].q[:]),collect(prefs[2].p[:]))]
        plot!(p[5],0:h:plotrange,ham,label= prefs[1],xaxis="time",yaxis="Hamiltonian")
        # scatter!(p[5],0:h:plotrange,ham,label="",markersize = 1,ylims = (-60,-40))
    end
    return p
end