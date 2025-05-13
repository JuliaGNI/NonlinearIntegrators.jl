using Pkg

# cd("IntegratorNN/GeometricIntegrators.jl")
# cd("..")
cd("IntegratorNN")

Pkg.activate(".")

using GeometricIntegrators
using NonlinearIntegrators
using QuadratureRules
using CompactBasisFunctions
using GeometricProblems: HarmonicOscillator
using GeometricProblems


# Set up the Harmonic Oscillator problem
int_step = 0.5
int_timespan = 5.

HO_iode = GeometricProblems.HarmonicOscillator.lodeproblem(tspan = (0,int_timespan),tstep = int_step)
HO_pref = HarmonicOscillator.exact_solution(HarmonicOscillator.podeproblem(tspan = (0,int_timespan),tstep = int_step))

#set up the Coupled Harmonic Oscillator problem
CHO = GeometricProblems.CoupledHarmonicOscillator.lodeproblem(tstep=int_step,tspan=(0,int_timespan))

QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
BGau4 = CompactBasisFunctions.Lagrange(QuadratureRules.nodes(QGau4))
CHO_pref = integrate(CHO, CGVI(BGau4, QGau4))

#set up the OuterSolarSystem
OSS = GeometricProblems.OuterSolarSystem.lodeproblem(tstep=int_step,tspan=(0,int_timespan),n=3)
OSS_pref = integrate(OSS, CGVI(BGau4, QGau4))




S₁ = 4
S = 4
square(x) = x^2
OLnetwork = DenseNet_GML{Float64}(tanh,S₁,S)
QGau4 = QuadratureRules.GaussLegendreQuadrature(4)
L_DenseGML = Linear_DenseNet_GML(OLnetwork,QGau4)


#HarmonicOscillator
HO_NLOLsol = integrate(HO_iode, L_DenseGML)
relative_maximum_error(HO_NLOLsol.q,HO_pref.q)


#CoupledHarmonicOscillator
CHO_NLOLsol = integrate(CHO, L_DenseGML)
relative_maximum_error(CHO_NLOLsol.q,CHO_pref.q)

#OuterSolarSystem
OSS_NLOLsol = integrate(OSS, L_DenseGML)
relative_maximum_error(OSS_NLOLsol.q,OSS_pref.q)



using GeometricIntegrators
using GeometricSolutions
using GeometricProblems.HarmonicOscillator

# number of samples, range of initial values
const nsamples = 6
const zmin = -0.6
const zmax = 0.6

# time step, number of time steps, integration time interval
const Δt = 0.01
const nt = 100
const tspan = (0.0, Δt * nt)


# create an ODE ensemble with equidistant sampling of the phasespace
ode = odeensemble([zmin, zmin], [zmax, zmax], [nsamples, nsamples]; tspan=tspan, tstep=Δt)

# integrate ODE ensemble using the implicit midpoint method
sol = integrate(ode, ImplicitMidpoint())

# convert the resulting solution into a snapshot matrix data.q
data = GeometricSolutions.arrays(sol)
data_t = repeat(collect(sol.t),nsamples^2)
data_t = reshape(data_t, 1,:)
data_q = reshape(data.q[1,:], 1, :)

using AbstractNeuralNetworks
using GeometricMachineLearning
nn = NeuralNetwork(Chain(Dense(1, 10, tanh), Dense(10, 10, tanh),Dense(10, 1)))

const batch_size = 72
const n_epochs = 500
opt = GeometricMachineLearning.Optimizer(AdamOptimizerWithDecay(nepochs, 1e-3, 5e-5), nn)

data_loader = dl_test = DataLoader(data_t, data_q)
batch = Batch(batch_size, data_loader)
backend = GeometricMachineLearning.networkbackend(data_loader)
loss_array1 = opt(nn, data_loader, batch, n_epochs, GeometricMachineLearning.FeedForwardLoss())


function mse_loss(x,y::AbstractArray{T},NN,ps) where T
    y_pred = NN(x,ps)
    mse_loss = mean(abs,y_pred - y)
    return mse_loss
end


err = 0
for ep in 1:nepochs
    gs = Zygote.gradient(p -> mse_loss(data_t, data_q, nn, p)[1], ps[k])[1]
    optimization_step!(opt, nn, ps[k], gs)
    err = mse_loss(network_inputs, labels, NN, ps[k])[1]
end