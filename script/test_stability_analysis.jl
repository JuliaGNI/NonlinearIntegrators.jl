

using Symbolics
using LinearAlgebra

"""
Generic derivation engine for variational integrator stability matrices
Parameters:
    s: polynomial order (P_s)
    r: number of quadrature points (N_r)
    quad_type: :Gauss or :Lobatto
"""
function derive_stability_matrix(s, r, quad_type)
    # 1. Define basic symbolic variables
    @variables h ω τ
    
    # Fix: manually create a scalar variable array to avoid Symbolics array indexing errors
    q_vars = [Symbolics.variable(:q, i) for i in 0:s]
    
    # 2. Determine quadrature nodes c and weights b (keep symbolic)
    # Use Num(5) to keep sqrt(5) symbolic
    c, b = if quad_type == :Gauss
        if r == 1; ([1//2], [1])
        elseif r == 2; ([1//2 - sqrt(Num(3))/6, 1//2 + sqrt(Num(3))/6], [1//2, 1//2])
        elseif r == 3; ([1//2 - sqrt(Num(15))/10, 1//2, 1//2 + sqrt(Num(15))/10], [5//18, 8//18, 5//18])
        else error("Need to manually add Gauss nodes for order $r") end
    elseif quad_type == :Lobatto
        if r == 2; ([0, 1], [1//2, 1//2])
        elseif r == 3; ([0, 1//2, 1], [1//6, 4//6, 1//6])
        elseif r == 4; ([0, (5-sqrt(Num(5)))/10, (5+sqrt(Num(5)))/10, 1], [1//12, 5//12, 5//12, 1//12])
        else error("Need to manually add Lobatto nodes for order $r") end
    end

    # 3. Build Lagrange basis functions (control points d are equally spaced)
    # Control points could also be chosen as quadrature nodes
    d = collect(range(0, 1, length=s+1))
    function L_basis(j, t)
        res = 1.0
        for m in 1:s+1
            m != j && (res *= (t - d[m]) / (d[j] - d[m]))
        end
        return res
    end

    # 4. Construct the discrete Lagrangian Ld
    q_poly = sum(q_vars[j] * L_basis(j, τ) for j in 1:s+1)
    q_dot_poly = Symbolics.derivative(q_poly, τ) # dq/dτ

    Ld = 0.0
    for i in 1:r
        # L = 1/2*v^2 - 1/2*ω^2*q^2, dt = h * dτ
        qi = substitute(q_poly, Dict(τ => c[i]))
        vi = (1/h) * substitute(q_dot_poly, Dict(τ => c[i]))
        Ld += h * b[i] * ( (1//2)*vi^2 - (1//2)*ω^2*qi^2 )
    end

    # 5. Extract linear system coefficients (second derivatives yield A matrix)
    # Ld = 1/2 * q' * A * q
    A = [Symbolics.derivative(Symbolics.derivative(Ld, q_vars[i]), q_vars[j]) for i in 1:s+1, j in 1:s+1]

    # 6. Elimination process (Schur complement on block matrix)
    # Block partition of A
    # [1] is q0, [2:s] are internal nodes, [s+1] is qs
    if s > 1
        idx_int = 2:s
        A_00 = A[1, 1]
        A_0I = A[1, idx_int]
        A_0s = A[1, s+1]
        
        A_II = A[idx_int, idx_int]
        A_I0 = A[idx_int, 1]
        A_Is = A[idx_int, s+1]
        
        A_s0 = A[s+1, 1]
        A_sI = A[s+1, idx_int]
        A_ss = A[s+1, s+1]

        invA_II = inv(A_II)
        
        # p0 = -D1 Ld, ps = Ds+1 Ld
        # Effective coefficients after enforcing internal stationarity
        X = -(A_00 - (A_0I' * invA_II * A_I0))
        Y = -(A_0s - (A_0I' * invA_II * A_Is))
        Z = (A_s0 - (A_sI' * invA_II * A_I0))
        W = (A_ss - (A_sI' * invA_II * A_Is))
    else
        # Linear interpolation: s=1 has no internal nodes
        X = -A[1, 1]
        Y = -A[1, 2]
        Z = A[2, 1]
        W = A[2, 2]
    end

    # 7. Build the state transition matrix (map [p0, ωq0] -> [ps, ωqs])
    m11 = simplify(W / Y)
    m12 = simplify((Z - (W * X) / Y) / ω)
    m21 = simplify(ω / Y)
    m22 = simplify(-X / Y)

    return [m11 m12; m21 m22]
end

# --- Tests and reproduction ---

println("1. (P1N1Q2Gau) Midpoint Rule:")
M1 = derive_stability_matrix(1, 1, :Gauss)
display(M1)

println("\n2. (P1N2Q2Lob) Trapezoidal Rule (Störmer-Verlet):")
M2 = derive_stability_matrix(1, 2, :Lobatto)
display(M2)

println("\n3. (P2N3Q4Lob) Example from paper p.17 (denominator includes 48):")
M3 = derive_stability_matrix(2, 3, :Lobatto)
display(M3)

println("\n4. (P3N4Q6Lob) Most complex example from paper p.18:")
M4 = derive_stability_matrix(3, 4, :Lobatto)
# This term is very long; simplify may take time
display(M4)

using CairoMakie
using LinearAlgebra
using FastGaussQuadrature

# --- 1. Numerical core: generate update matrix ---
function compute_matrix_numeric(s, r, quad_type, hw)
    h = 1.0
    omega = hw
    
    # Quadrature nodes and weights
    nodes, weights = quad_type == :Gauss ? gausslegendre(r) : gausslobatto(r)
    c = (nodes .+ 1) ./ 2
    b = weights ./ 2
    d = collect(range(0, 1, length=s+1))

    # Basis functions and derivatives
    l_basis(j, t) = prod((t - d[m])/(d[j] - d[m]) for m in 1:s+1 if m != j)
    function l_deriv(j, t)
        val = 0.0
        for m in 1:s+1
            if m != j
                term = 1.0/(d[j]-d[m])
                for k in 1:s+1
                    if k != j && k != m
                        term *= (t-d[k])/(d[j]-d[k])
                    end
                end
                val += term
            end
        end
        return val
    end

    # Build stiffness K and mass M
    K = [sum(b[k] * l_deriv(i, c[k]) * l_deriv(j, c[k]) for k in 1:r) for i in 1:s+1, j in 1:s+1]
    M = [sum(b[k] * l_basis(i, c[k]) * l_basis(j, c[k]) for k in 1:r) for i in 1:s+1, j in 1:s+1]
    
    A = (1/h)*K - (h*omega^2)*M

    # Elimination (Schur complement)
    if s > 1
        idx_int = 2:s
        # Note: for some hw, A_II can be singular; add a tiny perturbation or use pinv
        invA_II = inv(A[idx_int, idx_int] + I*1e-14) 
        X = -(A[1,1] - A[1,idx_int]' * invA_II * A[idx_int, 1])
        Y = -(A[1,s+1] - A[1,idx_int]' * invA_II * A[idx_int, s+1])
        Z = (A[s+1,1] - A[s+1,idx_int]' * invA_II * A[idx_int, 1])
        W = (A[s+1,s+1] - A[s+1,idx_int]' * invA_II * A[idx_int, s+1])
    else
        X, Y, Z, W = -A[1,1], -A[1,2], A[2,1], A[2,2]
    end

    # Map [p, omega*q]
    return [W/Y  (Z-W*X/Y)/omega; omega/Y  -X/Y]
end

# --- 2. Data preparation ---
function get_stability_data(s, r, quad_type, hw_range)
    λ_max = Float64[]
    λ_min = Float64[]
    for hw in hw_range
        temp_hw = abs(hw) < 1e-8 ? 1e-8 : hw
        M = compute_matrix_numeric(s, r, quad_type, temp_hw)
        evs = abs.(eigvals(M))
        push!(λ_max, maximum(evs))
        push!(λ_min, minimum(evs))
    end
    return λ_max, λ_min
end

# --- 3. Plotting ---
fig = Figure(resolution = (1000, 800), font = "DejaVu Sans")

# Subplot 1: Fig 2 (P2N3Q4Lob)
ax1 = Axis(fig[1, 1], title = "Fig 2: (P2N3Q4Lob)", xlabel = "hw", ylabel = "|λ|")
hw1 = range(-5, 5, length=500)
lmax1, lmin1 = get_stability_data(2, 3, :Lobatto, hw1)
lines!(ax1, hw1, lmax1, color = :red, label = "|λ₁|")
lines!(ax1, hw1, lmin1, color = :blue, label = "|λ₂|")
hlines!(ax1, [1.0], color = :black, linestyle = :dash)
axislegend(ax1)

# Subplot 2: Fig 3 (P3N4Q6Lob)
ax2 = Axis(fig[1, 2], title = "Fig 3: (P3N4Q6Lob)", xlabel = "hw")
hw2 = range(-8, 8, length=1000)
lmax2, lmin2 = get_stability_data(3, 4, :Lobatto, hw2)
lines!(ax2, hw2, lmax2, color = :red)
lines!(ax2, hw2, lmin2, color = :blue)
hlines!(ax2, [1.0], color = :black, linestyle = :dash)

# Subplot 3: Fig 4 (P3N4Q6Lob) zoom on unstable bubble
ax3 = Axis(fig[2, 1], title = "Fig 4: Zoom (P3N4Q6Lob)", xlabel = "hw", ylabel = "|λ|")
hw3 = range(3.1, 3.18, length=500)
lmax3, lmin3 = get_stability_data(3, 4, :Lobatto, hw3)
lines!(ax3, hw3, lmax3, color = :red)
lines!(ax3, hw3, lmin3, color = :blue)
hlines!(ax3, [1.0], color = :black, linestyle = :dash)
ylims!(ax3, 0.98, 1.02)

# Subplot 4: Gauss comparison (P3N3Q6Gau) shows A-stability
ax4 = Axis(fig[2, 2], title = "A-stable: (P3N3Q6Gau)", xlabel = "hw")
hw4 = range(-8, 8, length=500)
lmax4, lmin4 = get_stability_data(3, 3, :Gauss, hw4)
lines!(ax4, hw4, lmax4, color = :red)
lines!(ax4, hw4, lmin4, color = :blue)
hlines!(ax4, [1.0], color = :black, linestyle = :dash)
ylims!(ax4, 0.5, 1.5) # Show it stays 1 even over a large range

# Adjust layout spacing
colsize!(fig.layout, 1, Relative(0.5))
rowgap!(fig.layout, 30)
colgap!(fig.layout, 30)

# Save figure
save("stability_reproduction.png", fig)
display(fig)


using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature

"""
Compute the update matrix (Jacobian) and eigenvalues for a neural variational integrator
Parameters:
    s: number of neurons
    act: activation function (e.g., tanh, sin, sigmoid)
    w: first-layer weight vector (length s)
    b: first-layer bias vector (length s)
    h: time step size
    omega: oscillator frequency ω
    quad_order: number of quadrature points (r)
"""
function compute_neural_stability(s, act, w, b, h, omega, quad_order)
    # 1. Automatic differentiation: derivative of activation
    sigma(x) = act(x)
    sigma_prime(x) = ForwardDiff.derivative(sigma, x)

    # 2. Get quadrature nodes (Gauss-Legendre) and map to [0, 1]
    nodes, weights = gausslegendre(quad_order)
    c = (nodes .+ 1) ./ 2
    b_q = weights ./ 2

    # 3. Build basis functions phi_i(tau) = sigma(w_i * tau + b_i)
    # and derivatives dphi_i(tau) = w_i * sigma_prime(w_i * tau + b_i)
    phi(i, tau) = sigma(w[i] * tau + b[i])
    dphi(i, tau) = w[i] * sigma_prime(w[i] * tau + b[i])

    # 4. Build stiffness K and mass M in the basis space (s x s)
    # L_d = (1/2h) * alpha' * K * alpha - (h*omega^2/2) * alpha' * M * alpha
    K_mat = zeros(s, s)
    M_mat = zeros(s, s)

    for i in 1:s, j in 1:s
        for k in 1:quad_order
            # Quadrature sampling
            K_mat[i, j] += b_q[k] * dphi(i, c[k]) * dphi(j, c[k])
            M_mat[i, j] += b_q[k] * phi(i, c[k]) * phi(j, c[k])
        end
    end

    # Total energy matrix A_alpha
    A_alpha = (1/h) * K_mat - (h * omega^2) * M_mat

    # 5. Boundary constraints
    # Build constraint matrix C (2 x s): [q(0); q(1)] = C * alpha
    C = zeros(2, s)
    for j in 1:s
        C[1, j] = phi(j, 0.0)
        C[2, j] = phi(j, 1.0)
    end

    # 6. Eliminate internal degrees via variational principle (constrained extremum)
    # We need the Hessian of Ld w.r.t. boundary q = [q0, q1], denoted H_q
    # From constrained optimality: H_q = (C * A_alpha^-1 * C')^-1
    # This describes how energy changes with boundary q through optimal alpha*
    try
        # Invert A_alpha (if ill-conditioned, basis choice is poor)
        invA = inv(A_alpha)
        # Compute boundary Hessian
        H_q = inv(C * invA * C')

        # Extract Hessian components
        L00 = H_q[1, 1]
        L01 = H_q[1, 2]
        L10 = H_q[2, 1]
        L11 = H_q[2, 2]

        # 7. Build Jacobian update matrix J
        # Map [q0; p0] -> [q1; p1]
        # Note: standard variational integrator form p0 = -D1 Ld, p1 = D2 Ld
        J11 = -L00 / L01
        J12 = -1 / L01
        J21 = L10 - L11 * (L00 / L01)
        J22 = -L11 / L01

        J = [J11 J12; J21 J22]

        # 8. Compute eigenvalues
        vals = eigvals(J)

        return (Jacobian = J, Eigenvalues = vals, Determinant = det(J))

    catch e
        return "Matrix is singular: check whether basis functions (w, b) are linearly independent or quadrature order is sufficient."
    end
end

# --- Example run ---

# Parameters
s = 4                   # 4 neurons
act = tanh              # use tanh activation
w_rand = [2.0, -1.5, 3.0, 0.5]  # random weights
b_rand = [0.1, -0.2, 0.5, 0.0]  # random biases
h = 0.1                 # step size
omega = 1.0             # frequency
r = 6                   # 6-point Gauss quadrature

result = compute_neural_stability(s, act, w_rand, b_rand, h, omega, r)

if typeof(result) != String
    println("Update matrix (Jacobian):")
    display(result.Jacobian)
    println("\nEigenvalues:")
    display(result.Eigenvalues)
    println("\nEigenvalue magnitudes: ", abs.(result.Eigenvalues))
    println("\nDeterminant (symplectic check, should be near 1.0): ", result.Determinant)
else
    println(result)
end

using LinearAlgebra
using ForwardDiff
using QuadratureRules
using CairoMakie
using Printf

"""
Full stability analysis for a neural variational integrator
Considers sensitivity to all parameters (w, b, alpha)
"""
function compute_full_neural_stability(s, act, theta_0, h, omega, r)
    # theta_0 are parameters at the equilibrium point [w...; b...; alpha...]
    # s is the number of neurons; total parameters are typically 3s
    
    # 1. Define neural-network trajectory q(t, theta)
    function nntraj(t_normalized, p)
        w = zeros(eltype(p),s)
        b = zeros(eltype(p),s)
        α = zeros(eltype(p),s)
        k = 1
        D = 1 
        for i in 1:s
            α[i] = p[D*(i-1)+k]
            w[i] = p[D*(s+1)+D*(i-1)+k]
            b[i] = p[D*(s+1+s)+D*(i-1)+k]
        end

        # w = p[1:s]
        # b = p[s+1:2s]
        # α = p[2s+1:3s]
        # Map to [0, 1]
        return sum(α[i] * act(w[i] * t_normalized + b[i]) for i in 1:s)
    end

    # 2. Quadrature setup
    quad_rule = QuadratureRules.GaussLegendreQuadrature(r)
    b_q, c = quad_rule.weights, quad_rule.nodes


    # 3. Compute sensitivity basis functions Psi(t) = dq/dtheta
    # Evaluate Psi and Psi_dot at quadrature points
    num_p = length(theta_0)
    M_total = zeros(num_p, num_p)
    K_total = zeros(num_p, num_p)

    for k in 1:r
        tau = c[k]
        # Use automatic differentiation to compute gradient w.r.t. p (Psi)
        psi = ForwardDiff.gradient(p -> nntraj(tau, p), theta_0)
        
        # Compute Psi_dot (gradient of time-derivative w.r.t. p)
        # Use: d/dt (dq/dp) = d/dp (dq/dt)
        psi_dot = ForwardDiff.gradient(p -> ForwardDiff.derivative(t -> nntraj(t, p), tau), theta_0)

        # Build energy matrices
        M_total += b_q[k] * (psi * psi')
        K_total += b_q[k] * (psi_dot * psi_dot')
    end

    # Full system matrix (parameter space)
    # Note: K_total corresponds to (dq/dtau)^2; actual velocity is (1/h)*dq/dtau, so coefficient is 1/h
    A_theta = (1/h) * K_total - (h * omega^2) * M_total

    # 4. Build constraint matrix C (2 x num_p)
    C = zeros(2, num_p)
    C[1, :] = ForwardDiff.gradient(p -> nntraj(0.0, p), theta_0)
    C[2, :] = ForwardDiff.gradient(p -> nntraj(1.0, p), theta_0)

    # 5. Compute Hessian (projected to boundary space)
    # A_theta can be singular (parameter redundancy), so use pseudoinverse or regularization
    try
        # For neural networks, parameters are often redundant; A_theta may be rank-deficient
        # Use pinv (pseudoinverse) to handle this
        H_q = inv(C * pinv(A_theta) * C')

        # 6. Build Jacobian matrix J
        L00, L01 = H_q[1, 1], H_q[1, 2]
        L10, L11 = H_q[2, 1], H_q[2, 2]

        J = [-L00/L01  -1/L01; L10-L11*L00/L01  -L11/L01]
        
        return (Jacobian = J, Eigenvalues = eigvals(J), Det = det(J), Eigenvalues_norm = abs.(eigvals(J)))
    catch e
        return "Computation failed: $e"
    end
end

# --- Test run ---
m = 1.0
k = 0.5
omega = sqrt(k/m)
s = 4
r = 4
relu_k = 2
act = x->max(0.0,x) ^ relu_k
# Initial parameters: ensure the NN can express meaningful physics under these values
# For example, larger w implies higher frequency; alpha should not be all zeros
initial_x= [0.025330291766675752, 0.025330300250204125, -0.05191035724744225, 1.41857361323206e-6, -0.02497906796879972, 1.0, -1.0, 1.0, 1.0, 3.1415926535897922, 3.1415926535897922, -4.440892098500626e-16, -0.514702832400884]
final_x = [0.02533052411188818, 0.02533010942057078, -0.0519056811470894, 2.5769832077184494e-13, -0.024979171875433304, 0.999995335351156, -0.9999988884321463, 1.0000358570900132, 0.968858079436883, 3.1415912394269774, 3.141591481313382, 2.354529515720423e-5, -0.5659583990970317]


init_J = compute_full_neural_stability(s, act, initial_x, 0.1, omega, r)
final_J = compute_full_neural_stability(s, act, final_x, 0.1, omega, r)


function get_stability_data_full_neural(s, act, params, omega, r, hw_range)
    λ_max = Float64[]
    λ_min = Float64[]
    for hw in hw_range
        temp_hw = abs(hw) < 1e-8 ? 1e-8 : hw
        M = compute_full_neural_stability(s, act, params, temp_hw, omega, r)
        push!(λ_max, maximum(M.Eigenvalues_norm))
        push!(λ_min, minimum(M.Eigenvalues_norm))
    end
    return λ_max, λ_min
end

# --- 3. Plotting ---
fig = Figure(size = (1000, 800), font = "DejaVu Sans")

# Subplot 1: Fig 2 (P2N3Q4Lob)
ax1 = Axis(fig[1, 1], title = "S$(s)R$(r)k$(relu_k), Initial X", xlabel = "h", ylabel = "|λ|",yscale=log10)

y_min, y_max = 1 - 5e-13, 1 + 5e-13
ylims!(ax1, y_min, y_max)
ax1.ytickformat = values -> map(v -> v == 1.0 ? "1.0" : @sprintf("1 + %.0e", v - 1.0), values)

h_range = range(-0.1, 3.0, length=500)
hw1 = h_range .* omega
lmax1, lmin1 = get_stability_data_full_neural(s, act, initial_x, omega, r, hw1)
lines!(ax1, h_range, lmax1, color = :red, label = "|λ₁|")
lines!(ax1, h_range, lmin1, color = :blue, label = "|λ₂|",linestyle=:dash)
hlines!(ax1, [1.0], color = :black, linestyle = :dash)
axislegend(ax1)

fig
