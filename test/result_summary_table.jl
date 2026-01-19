using JLD2
using CairoMakie
using Statistics
using Colors

R_list = [8,16,4]
S_list = [4,6,8]
k_list = [2,3,4]
h_list = [0.1,0.2,0.5,1.0,2.0,5.0]
f_abs_list = [2.0,8.0]
x_abs_list = [2.0,8.0]

HO_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(f_abs_list),length(x_abs_list))
q_err_tensor = zeros(length(h_list),length(S_list),length(k_list),length(R_list),length(f_abs_list),length(x_abs_list))
for (hi,int_step) in enumerate(h_list)
    for (Si,S) in enumerate(S_list)
        for (ki,k_relu) in enumerate(k_list)
            for (Ri,R) in enumerate(R_list)
                Q = 2 * R
                for (fi,f_abs) in enumerate(f_abs_list)
                    for (xi,x_abs) in enumerate(x_abs_list)
                        data_file="default_linesearch/NVI_HO_h$(int_step)S$(S)R$(R)reluk=$(k_relu)fabs$(f_abs)xabs$(x_abs).jld2"
                        isfile(data_file) ? nothing : continue
                        result_data = load(data_file)
                        HO_err_tensor[hi,Si,ki,Ri,fi,xi] = result_data["HO_max_hams_err"]
                        q_err_tensor[hi,Si,ki,Ri,fi,xi] = result_data["HO_qerror"] 
                    end
                end
            end

        end
    end
end

fig = Figure(resolution = (1200, 800))
ax = Axis(fig[1, 1], xlabel = "Time Step h", ylabel = "HO Max Hamiltonian Error", 
    xscale = log10, yscale = log10, title = "Harmonic Oscillator Max Hamiltonian Error")
num_lines = length(S_list) * length(k_list)

line_colors = distinguishable_colors(num_lines)
line_idx = 1

# fix S,k and compute the mean,max,min over R,f_abs,x_abs
for (Si, S) in enumerate(S_list)
    for (ki, k_relu) in enumerate(k_list)
        err_mean = zeros(length(h_list))
        err_max = zeros(length(h_list))
        err_min = zeros(length(h_list))
        
        for (hi, h) in enumerate(h_list)
            errors = HO_err_tensor[hi, Si, ki, :, :, :]
            if isempty(filter(!isnan, errors)) 
                println("Skipping h=$(h), S=$(S), k=$(k_relu) due to all NaNs")
                continue
            end

            err_mean[hi] = mean(filter(!isnan, errors))
            err_max[hi] = maximum(filter(!isnan, errors))
            err_min[hi] = minimum(filter(!isnan, errors))
        end
        
        errlow = err_mean .- err_min
        errhigh = err_max .- err_mean
        scatterlines!(ax, h_list, err_mean, label="S$(S)k$(k_relu)", color=line_colors[line_idx],
                      markersize=6, linewidth=2)
        errorbars!(ax, h_list, err_mean, errlow, errhigh, linewidth=2, color=line_colors[line_idx],whiskerwidth = 10)
        line_idx += 1
    end
end

axislegend(ax, position=:lt)
fig
save("HO_hamiltonian_error.pdf", fig)
