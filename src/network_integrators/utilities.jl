#######
#General functions that are used in the network integrators
#######

function first_order_central_difference(f,x;ϵ=0.00001)
    return (f(x+ϵ)-f(x-ϵ))/(2*ϵ)
end

function mse_loss(x,y,NN,ps;λ=1000)
    y_pred = NN(x,ps)
    mse_loss = mean(abs,y_pred - y) + λ*abs2(y_pred[1]-y[1])
    return mse_loss
end
