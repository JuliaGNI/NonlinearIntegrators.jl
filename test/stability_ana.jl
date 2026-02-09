# using Symbolics
# using LinearAlgebra

# """
# 通用变分积分器稳定性矩阵推导引擎
# 参数:
#     s: 多项式阶数 (P_s)
#     r: 积分点数 (N_r)
#     quad_type: :Gauss 或 :Lobatto
# """
# function derive_stability_matrix(s, r, quad_type)
#     # 1. 定义基本符号变量
#     @variables h ω τ
    
#     # 修正点：手动创建标量变量数组，避免 Symbolics 符号数组索引错误
#     q_vars = [Symbolics.variable(:q, i) for i in 0:s]
    
#     # 2. 确定积分点 c 和权重 b (保持符号化)
#     # 使用 Num(5) 来确保 sqrt(5) 保持符号形态
#     c, b = if quad_type == :Gauss
#         if r == 1; ([1//2], [1])
#         elseif r == 2; ([1//2 - sqrt(Num(3))/6, 1//2 + sqrt(Num(3))/6], [1//2, 1//2])
#         elseif r == 3; ([1//2 - sqrt(Num(15))/10, 1//2, 1//2 + sqrt(Num(15))/10], [5//18, 8//18, 5//18])
#         else error("需要手动添加 $r 阶 Gauss 点") end
#     elseif quad_type == :Lobatto
#         if r == 2; ([0, 1], [1//2, 1//2])
#         elseif r == 3; ([0, 1//2, 1], [1//6, 4//6, 1//6])
#         elseif r == 4; ([0, (5-sqrt(Num(5)))/10, (5+sqrt(Num(5)))/10, 1], [1//12, 5//12, 5//12, 1//12])
#         else error("需要手动添加 $r 阶 Lobatto 点") end
#     end

#     # 3. 构造 Lagrange 基函数 (控制点 d 取等距)
#     d = collect(range(0, 1, length=s+1))
#     function L_basis(j, t)
#         res = 1.0
#         for m in 1:s+1
#             m != j && (res *= (t - d[m]) / (d[j] - d[m]))
#         end
#         return res
#     end

#     # 4. 构造离散拉格朗日量 Ld
#     q_poly = sum(q_vars[j] * L_basis(j, τ) for j in 1:s+1)
#     q_dot_poly = Symbolics.derivative(q_poly, τ) # dq/dτ

#     Ld = 0.0
#     for i in 1:r
#         # L = 1/2*v^2 - 1/2*ω^2*q^2, dt = h * dτ
#         qi = substitute(q_poly, Dict(τ => c[i]))
#         vi = (1/h) * substitute(q_dot_poly, Dict(τ => c[i]))
#         Ld += h * b[i] * ( (1//2)*vi^2 - (1//2)*ω^2*qi^2 )
#     end

#     # 5. 提取线性系统系数 (提取二阶导数得到 A 矩阵)
#     # Ld = 1/2 * q' * A * q
#     A = [Symbolics.derivative(Symbolics.derivative(Ld, q_vars[i]), q_vars[j]) for i in 1:s+1, j in 1:s+1]

#     # 6. 消元过程 (基于分块矩阵的 Schur 补)
#     # A 矩阵分块
#     # [1] 是 q0, [2:s] 是内部点, [s+1] 是 qs
#     if s > 1
#         idx_int = 2:s
#         A_00 = A[1, 1]
#         A_0I = A[1, idx_int]
#         A_0s = A[1, s+1]
        
#         A_II = A[idx_int, idx_int]
#         A_I0 = A[idx_int, 1]
#         A_Is = A[idx_int, s+1]
        
#         A_s0 = A[s+1, 1]
#         A_sI = A[s+1, idx_int]
#         A_ss = A[s+1, s+1]

#         invA_II = inv(A_II)
        
#         # p0 = -D1 Ld, ps = Ds+1 Ld
#         # 代入内部极值条件后的等效系数
#         X = -(A_00 - (A_0I' * invA_II * A_I0))
#         Y = -(A_0s - (A_0I' * invA_II * A_Is))
#         Z = (A_s0 - (A_sI' * invA_II * A_I0))
#         W = (A_ss - (A_sI' * invA_II * A_Is))
#     else
#         # 线性插值 s=1 没有内部点
#         X = -A[1, 1]
#         Y = -A[1, 2]
#         Z = A[2, 1]
#         W = A[2, 2]
#     end

#     # 7. 构建状态转移矩阵 (映射 [p0, ωq0] -> [ps, ωqs])
#     m11 = simplify(W / Y)
#     m12 = simplify((Z - (W * X) / Y) / ω)
#     m21 = simplify(ω / Y)
#     m22 = simplify(-X / Y)

#     return [m11 m12; m21 m22]
# end

# # --- 测试与复现 ---

# println("1. (P1N1Q2Gau) 中点规则 (Midpoint Rule):")
# M1 = derive_stability_matrix(1, 1, :Gauss)
# display(M1)

# println("\n2. (P1N2Q2Lob) 梯形规则 (Störmer-Verlet):")
# M2 = derive_stability_matrix(1, 2, :Lobatto)
# display(M2)

# println("\n3. (P2N3Q4Lob) 论文第17页例子 (分母含48的那个):")
# M3 = derive_stability_matrix(2, 3, :Lobatto)
# display(M3)

# println("\n4. (P3N4Q6Lob) 论文第18页最复杂的例子:")
# M4 = derive_stability_matrix(3, 4, :Lobatto)
# # 这一项会非常长，simplify 可能需要时间
# display(M4)

# using CairoMakie
# using LinearAlgebra
# using FastGaussQuadrature

# # --- 1. 数值计算核心：生成更新矩阵 ---
# function compute_matrix_numeric(s, r, quad_type, hw)
#     h = 1.0
#     omega = hw
    
#     # 积分点与权重
#     nodes, weights = quad_type == :Gauss ? gausslegendre(r) : gausslobatto(r)
#     c = (nodes .+ 1) ./ 2
#     b = weights ./ 2
#     d = collect(range(0, 1, length=s+1))

#     # 基函数及其导数
#     l_basis(j, t) = prod((t - d[m])/(d[j] - d[m]) for m in 1:s+1 if m != j)
#     function l_deriv(j, t)
#         val = 0.0
#         for m in 1:s+1
#             if m != j
#                 term = 1.0/(d[j]-d[m])
#                 for k in 1:s+1
#                     if k != j && k != m
#                         term *= (t-d[k])/(d[j]-d[k])
#                     end
#                 end
#                 val += term
#             end
#         end
#         return val
#     end

#     # 构造刚度 K 和质量 M
#     K = [sum(b[k] * l_deriv(i, c[k]) * l_deriv(j, c[k]) for k in 1:r) for i in 1:s+1, j in 1:s+1]
#     M = [sum(b[k] * l_basis(i, c[k]) * l_basis(j, c[k]) for k in 1:r) for i in 1:s+1, j in 1:s+1]
    
#     A = (1/h)*K - (h*omega^2)*M

#     # 消元 (Schur Complement)
#     if s > 1
#         idx_int = 2:s
#         # 注意：对于某些hw，A_II可能是奇异的，加一个极小值扰动或使用pinv
#         invA_II = inv(A[idx_int, idx_int] + I*1e-14) 
#         X = -(A[1,1] - A[1,idx_int]' * invA_II * A[idx_int, 1])
#         Y = -(A[1,s+1] - A[1,idx_int]' * invA_II * A[idx_int, s+1])
#         Z = (A[s+1,1] - A[s+1,idx_int]' * invA_II * A[idx_int, 1])
#         W = (A[s+1,s+1] - A[s+1,idx_int]' * invA_II * A[idx_int, s+1])
#     else
#         X, Y, Z, W = -A[1,1], -A[1,2], A[2,1], A[2,2]
#     end

#     # 映射 [p, omega*q]
#     return [W/Y  (Z-W*X/Y)/omega; omega/Y  -X/Y]
# end

# # --- 2. 数据准备函数 ---
# function get_stability_data(s, r, quad_type, hw_range)
#     λ_max = Float64[]
#     λ_min = Float64[]
#     for hw in hw_range
#         temp_hw = abs(hw) < 1e-8 ? 1e-8 : hw
#         M = compute_matrix_numeric(s, r, quad_type, temp_hw)
#         evs = abs.(eigvals(M))
#         push!(λ_max, maximum(evs))
#         push!(λ_min, minimum(evs))
#     end
#     return λ_max, λ_min
# end

# # --- 3. 绘图流程 ---
# fig = Figure(resolution = (1000, 800), font = "DejaVu Sans")

# # 子图 1: Fig 2 (P2N3Q4Lob)
# ax1 = Axis(fig[1, 1], title = "Fig 2: (P2N3Q4Lob)", xlabel = "hw", ylabel = "|λ|")
# hw1 = range(-5, 5, length=500)
# lmax1, lmin1 = get_stability_data(2, 3, :Lobatto, hw1)
# lines!(ax1, hw1, lmax1, color = :red, label = "|λ₁|")
# lines!(ax1, hw1, lmin1, color = :blue, label = "|λ₂|")
# hlines!(ax1, [1.0], color = :black, linestyle = :dash)
# axislegend(ax1)

# # 子图 2: Fig 3 (P3N4Q6Lob)
# ax2 = Axis(fig[1, 2], title = "Fig 3: (P3N4Q6Lob)", xlabel = "hw")
# hw2 = range(-8, 8, length=1000)
# lmax2, lmin2 = get_stability_data(3, 4, :Lobatto, hw2)
# lines!(ax2, hw2, lmax2, color = :red)
# lines!(ax2, hw2, lmin2, color = :blue)
# hlines!(ax2, [1.0], color = :black, linestyle = :dash)

# # 子图 3: Fig 4 (P3N4Q6Lob) Zoom 观察不稳定气泡
# ax3 = Axis(fig[2, 1], title = "Fig 4: Zoom (P3N4Q6Lob)", xlabel = "hw", ylabel = "|λ|")
# hw3 = range(3.1, 3.18, length=500)
# lmax3, lmin3 = get_stability_data(3, 4, :Lobatto, hw3)
# lines!(ax3, hw3, lmax3, color = :red)
# lines!(ax3, hw3, lmin3, color = :blue)
# hlines!(ax3, [1.0], color = :black, linestyle = :dash)
# ylims!(ax3, 0.98, 1.02)

# # 子图 4: Gauss 格式对比 (P3N3Q6Gau) 展示 A-稳定性
# ax4 = Axis(fig[2, 2], title = "A-stable: (P3N3Q6Gau)", xlabel = "hw")
# hw4 = range(-8, 8, length=500)
# lmax4, lmin4 = get_stability_data(3, 3, :Gauss, hw4)
# lines!(ax4, hw4, lmax4, color = :red)
# lines!(ax4, hw4, lmin4, color = :blue)
# hlines!(ax4, [1.0], color = :black, linestyle = :dash)
# ylims!(ax4, 0.5, 1.5) # 展现即便范围很大，它也一直等于1

# # 调整整体间距
# colsize!(fig.layout, 1, Relative(0.5))
# rowgap!(fig.layout, 30)
# colgap!(fig.layout, 30)

# # 保存图片
# save("stability_reproduction.png", fig)
# display(fig)


using LinearAlgebra
using ForwardDiff
using FastGaussQuadrature

"""
计算神经网络变分积分器的更新矩阵 (Jacobian) 及其特征值
参数:
    s: 神经元个数
    act: 激活函数 (例如 tanh, sin, sigmoid)
    w: 神经网络第一层权重向量 (长度 s)
    b: 神经网络第一层偏置向量 (长度 s)
    h: 时间步长
    omega: 谐振子频率 ω
    quad_order: 积分点数 (r)
"""
function compute_neural_stability(s, act, w, b, h, omega, quad_order)
    # 1. 自动微分：获取激活函数的导数
    sigma(x) = act(x)
    sigma_prime(x) = ForwardDiff.derivative(sigma, x)

    # 2. 获取积分点 (Gauss-Legendre) 并映射到 [0, 1]
    nodes, weights = gausslegendre(quad_order)
    c = (nodes .+ 1) ./ 2
    b_q = weights ./ 2

    # 3. 构造基函数 phi_i(tau) = sigma(w_i * tau + b_i)
    # 以及导数 dphi_i(tau) = w_i * sigma_prime(w_i * tau + b_i)
    phi(i, tau) = sigma(w[i] * tau + b[i])
    dphi(i, tau) = w[i] * sigma_prime(w[i] * tau + b[i])

    # 4. 构造基函数空间的 刚度矩阵 K 和 质量矩阵 M (s x s)
    # L_d = (1/2h) * alpha' * K * alpha - (h*omega^2/2) * alpha' * M * alpha
    K_mat = zeros(s, s)
    M_mat = zeros(s, s)

    for i in 1:s, j in 1:s
        for k in 1:quad_order
            # 积分采样
            K_mat[i, j] += b_q[k] * dphi(i, c[k]) * dphi(j, c[k])
            M_mat[i, j] += b_q[k] * phi(i, c[k]) * phi(j, c[k])
        end
    end

    # 总能量矩阵 A_alpha
    A_alpha = (1/h) * K_mat - (h * omega^2) * M_mat

    # 5. 边界约束处理
    # 构造约束矩阵 C (2 x s): [q(0); q(1)] = C * alpha
    C = zeros(2, s)
    for j in 1:s
        C[1, j] = phi(j, 0.0)
        C[2, j] = phi(j, 1.0)
    end

    # 6. 利用变分原理消去内部自由度 (求约束极值)
    # 我们要计算 Ld 对边界 q = [q0, q1] 的 Hessian 矩阵 H_q
    # 根据受约束的极值理论：H_q = (C * A_alpha^-1 * C')^-1
    # 这描述了边界 q 变化时，系统能量如何随最优参数 alpha* 改变
    try
        # 计算 A_alpha 的逆 (如果是病态的，说明基函数选取得不好)
        invA = inv(A_alpha)
        # 计算边界层面的 Hessian
        H_q = inv(C * invA * C')

        # 提取 Hessian 分量
        L00 = H_q[1, 1]
        L01 = H_q[1, 2]
        L10 = H_q[2, 1]
        L11 = H_q[2, 2]

        # 7. 构造雅可比更新矩阵 J
        # 映射 [q0; p0] -> [q1; p1]
        # 注意：变分积分器的标准形式 p0 = -D1 Ld, p1 = D2 Ld
        J11 = -L00 / L01
        J12 = -1 / L01
        J21 = L10 - L11 * (L00 / L01)
        J22 = -L11 / L01

        J = [J11 J12; J21 J22]

        # 8. 计算特征值
        vals = eigvals(J)

        return (Jacobian = J, Eigenvalues = vals, Determinant = det(J))

    catch e
        return "矩阵奇异：请检查基函数(w, b)是否线性无关或积分点是否足够。"
    end
end

# --- 示例运行 ---

# 设定参数
s = 4                   # 4个神经元
act = tanh              # 使用 tanh 激活函数
w_rand = [2.0, -1.5, 3.0, 0.5]  # 随机权重
b_rand = [0.1, -0.2, 0.5, 0.0]  # 随机偏置
h = 0.1                 # 步长
omega = 1.0             # 频率
r = 6                   # 6点高斯积分

result = compute_neural_stability(s, act, w_rand, b_rand, h, omega, r)

if typeof(result) != String
    println("更新矩阵 (Jacobian):")
    display(result.Jacobian)
    println("\n特征值:")
    display(result.Eigenvalues)
    println("\n特征值模长: ", abs.(result.Eigenvalues))
    println("\n行列式 (辛性检查, 应接近1.0): ", result.Determinant)
else
    println(result)
end