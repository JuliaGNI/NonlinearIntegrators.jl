using JLD2

err = 100
for R in [8,16,32]
    for iter in [1000,10000]
        for fab in ["1.78e-15","4.44e-16"]
            for fsuc in ["1.78e-15","4.44e-16"]
                filename = "parallel_result_figures/Backtracking2_R$(R)_h5.00_iter$(iter)_fabs$(fab)_fsuc$(fsuc)_TT200.jld2"
                res = load(filename)
                if res["PerturbedPendulum_hams_err"] <err
                    err = res["PerturbedPendulum_hams_err"]
                    println("New optimal found: R=$(R), iter=$(iter), fabs=$(fab), fsuc=$(fsuc), err=$(err)")
                    println("File: $(filename)")
                    println("PerturbedPendulum_hams_err: ", res["PerturbedPendulum_hams_err"])
                end
            end
        end
    end
end



# For the Henon-Heiles system


function hamiltonian(q1,q2,p1,p2)
    λ = 1.0
    0.5 * (p1^2 + p2^2) + 0.5 * (q1^2 + q2^2) + λ * (q1^2 * q2 - q2^3 / 3) 
end

initial_ham = hamiltonian(HHq1[1], HHq2[1], HHp1[1], HHp2[1])
ham = hamiltonian.(HHq1, HHq2, HHp1, HHp2)


err = 100
time_span = 60.0
h = 1.0
for R in [8,16,32]
    for iter in [1000,10000]
        for fab in ["1.78e-15","4.44e-16"]
            for fsuc in ["1.78e-15","4.44e-16"]
                filename = "parallel_result_figures/Backtracking2_R$(R)_h$(h)0_iter$(iter)_fabs$(fab)_fsuc$(fsuc)_TT200.jld2"
                res = load(filename)
                HHq1 = res["HenonHeiles_PR_sol_q1"][1:Int(time_span/h)+1]
                HHq2 = res["HenonHeiles_PR_sol_q2"][1:Int(time_span/h)+1]
                HHp1 = res["HenonHeiles_PR_sol_p1"][1:Int(time_span/h)+1]
                HHp2 = res["HenonHeiles_PR_sol_p2"][1:Int(time_span/h)+1]
                initial_ham = hamiltonian(HHq1[1], HHq2[1], HHp1[1], HHp2[1])

                ham = hamiltonian.(HHq1, HHq2, HHp1, HHp2)
                HH_relative_hams_err = abs.((ham .- initial_ham) / initial_ham)
                max_err = maximum(HH_relative_hams_err)
                if max_err <err
                    err = max_err
                    println("New optimal found: R=$(R), iter=$(iter), fabs=$(fab), fsuc=$(fsuc), err=$(err)")
                    println("File: $(filename)")
                    println("HH_hams_err: ", max_err)
                    final_filename = filename
                end
            end
        end
    end
end


