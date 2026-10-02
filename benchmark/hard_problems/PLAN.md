# 计划：在"线性方法吃力"的 ODE 问题上比较非线性 NVI 与线性方法

目标：检验 `relu_cgvi_equivalence.md` §7 的结论"非线性 NVI 只可能在一步内有局部快速变化、频率未知且变化、或需要在固定步长下自适应的问题上胜出"。四个问题都是 Lagrangian ODE，可以写成 `LODEProblem`。

**判定标准（事先固定）。** 某个非线性方法在某个参数区间"更好"，当且仅当：

1. 在误差–代价图上，它在同等代价下比**所有**线性对照（L1、L2）的误差小 3 倍以上；
2. 它是稳健的：失败率不超过 5%，fallback 步数单独报告。

代价同时报告墙钟时间、Newton 迭代次数，以及（参考实现中的）FLOP 数。

---

## 1. 四个问题

| | 问题 | Lagrangian | 维数 | 守恒量 / 不变量 | 线性方法的难处 | 来源 |
|---|---|---|---|---|---|---|
| P1 | 频率缓变振子 | ½q̇² − ½ω(εt)²q²，ω(s) = 1 + ½ sin s | 1，**非自治** | H 不守恒；绝热不变量 J = H/ω | h 超过周期 2π/ω 后多项式欠分辨；固定频率的三角基会失配 | GeometricProblems.jl 新分支 |
| P2 | 非均匀强磁场中的带电粒子（2D，B 沿 z；给定静态场，粒子轨道是 ODE，不是 PDE） | ½m\|ẋ\|² + A(x)·ẋ − φ(x)，A、φ 取 `MasslessChargedParticle` 中的定义 | 2 | H；磁矩 μ = m v⊥²/(2B)（绝热）| 需要 h ≪ 2π m/B(x)，而 B(x) 沿轨道变化 | GeometricProblems.jl 新分支（m → 0 时退化为已有的 massless 问题，即导心极限）|
| P3 | 高偏心率 Kepler | ½\|q̇\|² + 1/\|q\| | 2 | H、角动量 L、Runge–Lenz 向量（进动误差）| 近日点附近一步内变化剧烈；固定步长浪费在远日点 | GeometricProblems.jl 新分支，有解析参考解（Kepler 方程）|
| P4 | Hénon–Heiles（准周期区，E < 1/6）| GeometricProblems `HenonHeilesPotential.lodeproblem` | 2 | H | 振幅缓变，两个分量之间交换能量 | 已有 |

参数（quick / full）：

- **P1**：ε ∈ {1e-2, 1e-3}，T = 2π/ε（一个调制周期），q₀ = 1，p₀ = 0。
- **P2**：A₀ = 1，E₀ ∈ {0, 0.01}，m ∈ {1e-1, 1e-2}（Ω 约为 10 和 100），T 取约 2 个漂移周期。
- **P3**：a = 1，e ∈ {0.9, 0.99}，T = 10 个轨道周期（2π 每个）。
- **P4**：两组初值，(q, p) = (0.1, 0.1, 0.1, 0.1)（论文 Table 4.4，E ≈ 0.02）和 (0.2, 0.2, 0.3, 0.3)（包默认值，E ≈ 0.135）。q 误差只在 T = 100 以内计算（混沌敏感性），ΔH 计算到 T = 1000。
- **步长**：按物理时间尺度归一化，h·Ω_max ∈ {0.1, 0.3, 1, 3, 10}。P3 用平均运动归一化；P1 和 P4 用 ω ≈ 1。

参考解：P3 用解析解；P1、P2、P4 用 Gauss(8)，dt 取最小步长的 1/50，并用 dt/2 自检。

## 2. 比较的方法

| 编号 | 方法 | 类型 | 实现 |
|---|---|---|---|
| L1 | CGVI(P_s, R = s+1)，s = 2..6，配合子步数 m = 1..4 | 线性，辛 | 包 |
| L2 | Gauss(s) Runge–Kutta | 线性，辛 | 包 |
| N1 | ShallowNet ReLU³，bias 为 kinkfree | 理论上等价于 CGVI(P₃)（定理 1）| 包；作为新问题上的健全性检查 |
| N2 | ShallowNet tanh / ReLU³ 默认 bias（OGA 会选到内部 kink）| 非线性 | 包 |
| N3 | VISE，用论文中的 ansatz（只做 P4，复现 Table 4.4）| 非线性 | 包 |
| N4 | 自由节点样条 VI | 非线性（节点），辛 | 包：新写 `src/spline/free_knot_spline.jl`（D ≥ 1，接口同 `ShallowNet`）；1D 时与 `theory/free_knot_vi.jl` 对照；主要用于 P3 |
| N5 | 自由频率 VI：ansatz Σⱼ (aⱼ cos ωhτ + bⱼ sin ωhτ) τʲ 加多项式，并加上条件 ∂L_d/∂ω = 0 | 非线性（频率），辛 | `theory`（新写，沿用 free-knot 的外层/内层结构）|

最关键的对照是 **N5 对 L1/L2**，以及 **N4 对 L1 的子步版本**。

频率 ω 固定的调制三角 Galerkin VI（同一 ansatz 但参数冻结的线性版本）本计划不做：它属于 GeometricIntegrators.jl，不属于 NVI，已记入 Obsidian（`GeometricIntegrators/`）。因此本计划只能说明 N5 是否优于标准线性方法；要断定收益来自非线性本身，需要等那个线性版本实现后再比。

## 3. 指标

- q 误差：网格上的最大相对误差（P4 只算到 T = 100）。
- 不变量：
  - P2、P3、P4：ΔH = maxₙ |Hₙ − H₀|/|H₀|；
  - P1：ΔJ；P2：Δμ（与参考解对比，因为 μ 本身只近似守恒）；
  - P3：ΔL 和近日点进动角误差。
- 代价：墙钟时间、Newton 迭代次数、失败率；参考实现另报 FLOP。
- 诊断：ShallowNet 每步用 `classify_step` 分类（P_k-equiv / spline / degenerate）；N4、N5 报告 fallback 步数。
- 图：每个问题两张。
  - 误差–代价 Pareto 图：横轴 FLOP 或墙钟时间，纵轴 q 误差；
  - 不变量误差–h 图：与 `relu_k_sweep` 的图同样风格，图例统一放在图下方。

## 4. 代码结构（尽量简单）

P1–P3 直接在 **GeometricProblems.jl 的新分支**（例如 `nvi-hard-problems`）里实现，按该包已有模块的风格写（参照 `harmonic_oscillator.jl`、`massless_charged_particle.jl`、`henon_heiles_potential.jl`）：

```
GeometricProblems.jl/                     # 新分支
  src/frequency_modulated_oscillator.jl   # P1：lagrangian、hamiltonian（含 t）、adiabatic_invariant、default_parameters、lodeproblem / hodeproblem
  src/charged_particle_2d.jl              # P2：A、φ 与 MasslessChargedParticle 相同，加质量 m；hamiltonian、magnetic_moment、lodeproblem / hodeproblem
  src/kepler_problem.jl                   # P3：lagrangian、hamiltonian、angular_momentum、runge_lenz_vector、exact_solution（Kepler 方程）、lodeproblem / hodeproblem
  src/GeometricProblems.jl                # include 并 export 新模块
  test/<problem>_tests.jl                 # 每个问题：lodeproblem 与 hodeproblem 用 Gauss(8) 积分结果一致；P3 与 exact_solution 一致到 1e-10
```

NonlinearIntegrators.jl 这边，`benchmark` 环境用 `Pkg.develop(path = "../../GeometricProblems.jl")`（假设该仓库与 NonlinearIntegrators.jl 同在 `GNI/` 下）指向该分支：

```
benchmark/hard_problems/
  setup.jl           # 四个问题的构造、参考解、不变量的统一接口（只调用 GeometricProblems，不重新定义问题）
  run_baselines.jl   # L1、L2 → results/<problem>_baselines.csv
  run_nvi.jl         # N1–N4（逐步积分 + classify_step，复用 relu_k_sweep.jl 的写法）→ results/<problem>_nvi.csv
  report.jl          # 读取 CSV，生成 Pareto 图、不变量图和 md 汇总
benchmark/theory/
  free_frequency_vi.jl  # N5，只用标准库，带 FLOP 计数，自检：辛性 |det − 1|、∂L_d/∂ω 与有限差分一致
src/spline/
  free_knot_spline.jl   # N4：FreeKnotSpline 方法，外层对节点求根、内层固定节点 Galerkin、复合求积跟随节点；
                        # 按 GeometricIntegratorsBase 的风格实现 method / cache / integrate_step!，在 NonlinearIntegrators.jl 中 include 并 export
test/
  free_knot_spline_smoke.jl  # 1D 时与 theory/free_knot_vi.jl 逐步一致；辛性 |det − 1|；m = 0 时等于 CGVI(P_k)
```

GeometricProblems.jl 的分支测试通过后，可以作为上游 PR 提交。

## 5. 阶段与顺序

| 阶段 | 内容 | 交付物 / 检查 |
|---|---|---|
| 0 | GeometricProblems.jl 新分支：三个新问题、参考解、不变量及其测试；benchmark 环境 develop 该分支 | P3 的数值解与解析解一致到 1e-10；P1 的 J 漂移随 ε 按 O(ε) 变化；P2 在 m 小时，导心轨迹与 `MasslessChargedParticle` 一致 |
| 1 | 线性基线 L1、L2（Boris 推进和 Sundman 变换暂不加入，见 Obsidian）| 每个问题的线性 Pareto 前沿 |
| 2 | 包里已有的 NVI（N1–N3）| N1 在新问题上仍与 CGVI(P₃) 相同（定理 1 的推广检查）；N2、N3 的结果和失败率 |
| 3 | N5 参考实现（P1 → P2 → P4）；N4 写进 src（P3）| 自检全部通过；N5 对 L1/L2、N4 对 L1 子步的对比 |
| 4 | 汇总 | `results/hard_problems.md`，加上每个问题的 Pareto 图；结论写进 `relu_cgvi_equivalence.md` 新增的一节 C8 |

建议的问题顺序：**P1 → P2 → P3 → P4**。

- P1 是一维问题，结构已知，最便宜，可以直接检验"自由频率"的想法。
- P2 和 IPP 的应用最相关。
- P3 检验自由节点。
- P4 有混沌，最难评价，放在最后。

## 6. 风险

- **N5 的混叠**：ω 和 ω + 2πn/h 都可能满足驻点条件。对策：ω 从上一步热启动，限制单步的变化量，和自由节点的做法一样。
- **ShallowNet 在 P3 近日点附近**：Newton 可能不收敛。如实记录失败率，不调参掩盖。
- **P2 的 Δμ 只是近似守恒**：必须和参考解的 μ(t) 对比，不能和常数比。
- **公平性**：包里的方法和参考实现的墙钟时间不能直接比。FLOP 只在同一实现族之内比较，跨族只比较误差–步长。
