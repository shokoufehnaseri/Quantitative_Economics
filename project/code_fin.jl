###########################################################
# Quantitative Economics Project:
# Tax Reform in a Heterogeneous Agent Model with Data Analysis
#
# This script analyzes the effects of a tax reform that
# increases labor tax progressivity (λ changes from 0 to 0.15)
# in a heterogeneous-agent model. It solves the household's
# Bellman equation using both a grid-based VFI and an
# interpolation-based VFI, simulates the stationary distribution,
# calibrates β to clear the asset market, and conducts data analysis
# (including regression and Lorenz curves) on the simulation outputs.
#
# Required packages:
#   Distributions, Statistics, Plots, Random, Interpolations
###########################################################

using Distributions
using Statistics
using Plots
using Random
using Interpolations
using StatsBase

##############################
# Parameter Structure
##############################
struct Params
    β::Float64      # discount factor
    γ::Float64      # risk aversion parameter
    τ::Float64      # tax parameter (set to meet government revenue)
    λ::Float64      # degree of tax progressivity
    r::Float64      # interest rate
    w::Float64      # wage rate
    ρ::Float64      # persistence of productivity shock
    σ::Float64      # standard deviation of productivity shock
end

##############################
# Tauchen's Method for Discretization
##############################
function tauchen(n::Int, mu::Float64, ρ::Float64, σ::Float64; m::Float64=3)
    
    z_std = sqrt(σ^2 / (1 - ρ^2))
    z_max = mu + m * z_std
    z_min = mu - m * z_std
    z_grid = collect(range(z_min, z_max, length=n))
    step = (z_max - z_min) / (n - 1)
    transition = zeros(n, n)
    for i in 1:n
        for j in 1:n
            if j == 1
                transition[i,j] = cdf(Normal(0, σ), (z_grid[1] - ρ * z_grid[i] + step/2) / σ)
            elseif j == n
                transition[i,j] = 1 - cdf(Normal(0, σ), (z_grid[n] - ρ * z_grid[i] - step/2) / σ)
            else
                lower = (z_grid[j] - ρ * z_grid[i] - step/2) / σ
                upper = (z_grid[j] - ρ * z_grid[i] + step/2) / σ
                transition[i,j] = cdf(Normal(0, σ), upper) - cdf(Normal(0, σ), lower)
            end
        end
    end
   
    z_grid_exp = exp.(z_grid)
    return z_grid_exp, transition
end

##############################
# Utility Function
##############################
function utility(c::Float64, γ::Float64)
    if c <= 0
        return -1e10  # Heavy penalty for non-positive consumption
    elseif abs(γ - 1) < 1e-8
        return log(c)
    else
        return (c^(1-γ) - 1) / (1-γ)
    end
end

##############################
# Standard Grid-based Value Function Iteration (VFI)
##############################
function solve_vfi(params::Params, grid_a::Vector{Float64}, grid_z::Vector{Float64}, transition::Array{Float64,2};
                   tol::Float64=1e-5, maxiter::Int=1000)
    n_a = length(grid_a)
    n_z = length(grid_z)
    V = zeros(n_a, n_z)
    policy = zeros(n_a, n_z)
    β, γ, τ, λ, r, w = params.β, params.γ, params.τ, params.λ, params.r, params.w

    # Iterate on the Bellman equation: V(a,z) = max_{a'} { u(c) + β E[V(a', z')] }
    for iter in 1:maxiter
        V_new = similar(V)
        for ia in 1:n_a
            a = grid_a[ia]
            for iz in 1:n_z
                z = grid_z[iz]
                max_val = -1e10
                a_opt = grid_a[1]
                
                disp_income = (1 - τ) * (w * z)^(1 - λ)
                for ia_prime in 1:n_a
                    a_prime = grid_a[ia_prime]
                    c = disp_income + (1 + r) * a - a_prime
                    u_val = utility(c, γ)
                    EV = 0.0
                    for iz_next in 1:n_z
                        EV += transition[iz, iz_next] * V[ia_prime, iz_next]
                    end
                    val = u_val + β * EV
                    if val > max_val
                        max_val = val
                        a_opt = a_prime
                    end
                end
                V_new[ia, iz] = max_val
                policy[ia, iz] = a_opt
            end
        end
        diff = maximum(abs.(V_new .- V))
        V = V_new
        println("Standard VFI iteration $iter, diff = $diff")
        if diff < tol
            break
        end
    end
    return V, policy
end

##############################
# Interpolation-based Value Function Iteration (VFI)
##############################
function solve_vfi_interp(params::Params, grid_a::Vector{Float64}, grid_z::Vector{Float64}, transition::Array{Float64,2};
                          tol::Float64=1e-5, maxiter::Int=1000)
    n_a = length(grid_a)
    n_z = length(grid_z)
    V = zeros(n_a, n_z)
    policy = zeros(n_a, n_z)
    β, γ, τ, λ, r, w = params.β, params.γ, params.τ, params.λ, params.r, params.w

    # Use linear interpolation to speed up VFI.
    for iter in 1:maxiter
        V_new = similar(V)
        
        interp_V = [LinearInterpolation(grid_a, V[:, iz], extrapolation_bc=Flat()) for iz in 1:n_z]
        
        for ia in 1:n_a
            a = grid_a[ia]
            for iz in 1:n_z
                z = grid_z[iz]
                max_val = -1e10
                a_opt = grid_a[1]
                disp_income = (1 - τ) * (w * z)^(1 - λ)
                for a_prime in grid_a
                    c = disp_income + (1 + r) * a - a_prime
                    u_val = utility(c, γ)
                    EV = 0.0
                    for iz_next in 1:n_z
                        EV += transition[iz, iz_next] * interp_V[iz_next](a_prime)
                    end
                    val = u_val + β * EV
                    if val > max_val
                        max_val = val
                        a_opt = a_prime
                    end
                end
                V_new[ia, iz] = max_val
                policy[ia, iz] = a_opt
            end
        end
        diff = maximum(abs.(V_new .- V))
        V = V_new
        println("Interpolation VFI iteration $iter, diff = $diff")
        if diff < tol
            break
        end
    end
    return V, policy
end

##############################
# Simulation of the Stationary Distribution
##############################
function simulate_stationary_distribution(policy::Array{Float64,2}, grid_a::Vector{Float64},
                                          grid_z::Vector{Float64}, transition::Array{Float64,2};
                                          simT::Int=20000, burnin::Int=2000)
    n_a = length(grid_a)
    n_z = length(grid_z)
    N = 20000  # number of households
    
    a_indices = ones(Int, N)
    z_indices = rand(1:n_z, N)
    
    for t in 1:simT
        for i in 1:N
            current_a_index = a_indices[i]
            current_z_index = z_indices[i]
            a_next = policy[current_a_index, current_z_index]
            
            new_a_index = searchsortedfirst(grid_a, a_next)
            a_indices[i] = min(new_a_index, n_a)
            
            prob = transition[current_z_index, :]
            z_indices[i] = sample(1:n_z, Weights(prob))
        end
    end
    asset_vals = grid_a[a_indices]
    return asset_vals, a_indices, z_indices
end

##############################
# Gini Coefficient Calculation
##############################
function gini(x::Vector{Float64})
    n = length(x)
    sorted_x = sort(x)
    idx = collect(1:n)
    return (2 * sum(idx .* sorted_x)) / (n * sum(sorted_x)) - (n + 1) / n
end

##############################
# Lorenz Curve Computation
##############################
function lorenz_curve(data::Vector{Float64})
    sorted_data = sort(data)
    n = length(sorted_data)
    cum_data = cumsum(sorted_data)
    total = sum(sorted_data)
    lorenz = [0.0; cum_data ./ total]
    population = collect(range(0, 1, length=length(lorenz)))
    return population, lorenz
end

##############################
# Regression Function (OLS)
##############################
function fit_regression(x::Vector{Float64}, y::Vector{Float64})
    β1 = cov(x, y) / var(x)
    β0 = mean(y) - β1 * mean(x)
    return β0, β1
end

##############################
# Data Analysis on Simulation Output
##############################
function analyze_simulation(asset_vals::Vector{Float64}, income_vals::Vector{Float64}, regime::String)
    correlation = cor(asset_vals, income_vals)
    println("Correlation between assets and after-tax income ($regime): ", round(correlation, digits=4))
    
    β0, β1 = fit_regression(asset_vals, income_vals)
    println("Regression ($regime): intercept = ", round(β0, digits=4), ", slope = ", round(β1, digits=4))
    
    sc = scatter(asset_vals, income_vals; label="Data ($regime)", color=:blue, markersize=2, alpha=0.6)
    xs = range(minimum(asset_vals), maximum(asset_vals), length=100)
    ys = [β0 + β1*x for x in xs]
    plot!(sc, xs, ys; label="Fitted line ($regime)", linewidth=3, color=:red)
    title!("Assets vs. After-tax Income ($regime)")
    xlabel!("Assets")
    ylabel!("After-tax Income")
    savefig(sc, "assets_income_scatter_$(regime).png")
end

##############################
# Average Simulation over Multiple Runs
##############################
function avg_simulation(policy, grid_a, grid_z, transition; runs::Int=3, simT::Int=10000, burnin::Int=1000)
    total_agg = 0.0
    for r in 1:runs
        asset_vals, a_idx, z_idx = simulate_stationary_distribution(policy, grid_a, grid_z, transition; simT=simT, burnin=burnin)
        total_agg += mean(asset_vals)
    end
    return total_agg / runs
end

##############################
# Beta Calibration Routine (with Averaging)
##############################
function calibrate_beta(beta_lower, beta_upper, tol_beta, grid_a, grid_z, transition, params_template, K_target; use_interp=true)
    max_iter = 1
    V, policy = nothing, nothing  
    params = nothing  
    agg_assets = 0.0

    for iter in 1:max_iter
        beta_mid = (beta_lower + beta_upper) / 2
        params = Params(beta_mid, params_template.γ, params_template.τ, params_template.λ,
                        params_template.r, params_template.w, params_template.ρ, params_template.σ)
                        
        # Solve household problem
        if use_interp
            V, policy = solve_vfi_interp(params, grid_a, grid_z, transition)
        else
            V, policy = solve_vfi(params, grid_a, grid_z, transition)
        end
        
        
        agg_assets = avg_simulation(policy, grid_a, grid_z, transition; runs=3, simT=10000, burnin=1000)
        diff = agg_assets - K_target
        println("Beta calibration iteration $iter: beta = $(beta_mid), agg_assets = $(agg_assets), diff = $(diff)")
        
        if abs(diff) < tol_beta
            return beta_mid, params, V, policy, agg_assets
        end
        
        
        if diff > 0  # Too high assets: agents too patient → lower beta.
            beta_upper = beta_mid
        else          # Too low assets: agents not patient enough → increase beta.
            beta_lower = beta_mid
        end
    end
    
   
    return (beta_lower + beta_upper) / 2, params, V, policy, agg_assets
end

#############################
# Find new Tau
#############################
function find_tau_progressive(params_baseline, λ_new, grid_a, grid_z, transition)
    τ_low, τ_high = 0.1, 0.5  
    tol = 1e-2
    max_iter = 5  # preventing infinite loops
    iter_count = 0  

    while abs(τ_high - τ_low) > tol
        iter_count += 1
        if iter_count > max_iter
            println("⚠️ Warning: Reached max iterations in `find_tau_progressive`")
            break
        end

        τ_mid = (τ_low + τ_high) / 2

        # Debug
        if τ_mid < 0 || τ_mid > 1
            println("❌ Error: Invalid τ_mid! Value = ", τ_mid)
            return NaN  
        end

        params_new = Params(params_baseline.β, params_baseline.γ, τ_mid, λ_new,
                            params_baseline.r, params_baseline.w, params_baseline.ρ, params_baseline.σ)

        _, policy = solve_vfi_interp(params_new, grid_a, grid_z, transition)
        asset_vals, _, z_idx = simulate_stationary_distribution(policy, grid_a, grid_z, transition)
        
        # Compute government revenue in new regime
        after_tax_income = [(1 - τ_mid) * (params_new.w * grid_z[z])^(1 - λ_new) for z in z_idx]
        G_new = sum(τ_mid * (params_new.w * grid_z[z]) for z in z_idx) / length(z_idx)

        # Debug
        if isnan(G_new) || G_new < 0
            println("❌ Error: Invalid government revenue! G_new = ", G_new, ", τ_mid = ", τ_mid)
            return NaN  # Signal failure
        end

        # Debug
        if iter_count % 1 == 0
            println("🔍 Iteration ", iter_count, ": τ_mid = ", round(τ_mid, digits=4), 
                    ", G_new = ", round(G_new, digits=4), ", Target G = ", params_baseline.τ)
        end

        
        if G_new > params_baseline.τ
            τ_high = τ_mid
        else
            τ_low = τ_mid
        end
    end

    println("✅ Converged: τ = ", round((τ_low + τ_high) / 2, digits=4))
    return (τ_low + τ_high) / 2
end


##############################
# Find new r and w
##############################
function find_equilibrium_prices(params, grid_a, grid_z, transition, α, A, δ)
    r_low, r_high = 0.01, 0.06  
    tol = 1e-4
    w_new = 0.0
    K_new = 0.0
    max_iter = 10 
    iter_count = 0  

    while abs(r_high - r_low) > tol
        iter_count += 1
        if iter_count > max_iter
            println("⚠️ Warning: Reached max iterations in `find_equilibrium_prices`")
            break
        end

        r_mid = (r_low + r_high) / 2
        w_new = (1 - α) * A * ((α * A) / (r_mid + δ))^(α / (1 - α))

        params_new = Params(params.β, params.γ, params.τ, params.λ, r_mid, w_new, params.ρ, params.σ)

        _, policy = solve_vfi_interp(params_new, grid_a, grid_z, transition)
        asset_vals, _, _ = simulate_stationary_distribution(policy, grid_a, grid_z, transition)
        K_new = mean(asset_vals)

        K_target = ((α * A) / (r_mid + δ))^(1 / (1 - α))

        if iter_count % 1 == 0
            println("🔍 Iteration ", iter_count, ": r_mid=", round(r_mid, digits=4), 
                    ", w_new=", round(w_new, digits=4), ", K_new=", round(K_new, digits=4),
                    ", K_target=", round(K_target, digits=4))
        end

        if K_new > K_target
            r_high = r_mid
        else
            r_low = r_mid
        end
    end

    println("✅ Converged: r =", round((r_low + r_high) / 2, digits=5), 
            ", w =", round(w_new, digits=5))
    return (r_low + r_high) / 2, w_new, K_new
end

#removed the wrapper function - not as pretty, but better for the dirty work
    ##############################
    # Main Function: Calibration, VFI, Simulation, Plots, Data Analysis
    ##############################
    # Set use_interp = true to use interpolation-based VFI.
function main()
    use_interp = true

    ##############################
    # Firm-side Calibration
    ##############################
    α = 0.36                     # U.S. labor share: 1 - α ≈ 0.64
    r_eq = 0.04                  # interest rate
    w_eq = 1.0                   # wage rate
    
    K = (α - 0.2) / (0.04 * (1 - α))
    println("Equilibrium Capital, K = ", K)
    A = w_eq / ((1 - α) * K^α)
    println("Equilibrium Productivity, A = ", A)
    δ = 0.2 * A * K^(α - 1)
    println("Equilibrium Depreciation, δ = ", δ)
    Y = A * K^α
    println("Output, Y = ", Y)
    τ_baseline = 0.2 * Y        # Tax parameter to meet G/Y = 0.2 in baseline
    println("Baseline tax parameter, τ = ", τ_baseline)

    # Other parameters
    γ = 2.0
    β_initial = 0.96          # Initial guess for discount factor
    ρ = 0.9
    σ = 0.4
    φ = 0.0                   # Borrowing constraint

    ##############################
    # Grids and Discretization
    ##############################
    n_a = 700
    a_min = φ
    a_max = 20.0
    grid_a = collect(range(a_min, a_max, length=n_a))

    n_z = 5    # Number of productivity states (per instructions)
    mu = 0.0   
    grid_z, transition = tauchen(n_z, mu, ρ, σ; m=3.0)

    ##############################
    # Beta Calibration: Adjust β so that simulated asset holdings equal K
    ##############################
    # Template parameters for calibration (using baseline tax and λ = 0)
    params_template = Params(β_initial, γ, τ_baseline, 0.0, r_eq, w_eq, ρ, σ)
    tol_beta = 1e-3
    beta_calibrated, params_cal, V_cal, policy_cal, agg_assets_cal = calibrate_beta(0.90, 0.98, tol_beta, grid_a, grid_z, transition, params_template, K; use_interp=use_interp)
    println("Calibrated beta: ", beta_calibrated)

    # updating baseline parameters with calibrated beta.
    params_baseline = Params(beta_calibrated, γ, τ_baseline, 0.0, r_eq, w_eq, ρ, σ)

    ##############################
    # Solve Household Problem: Baseline (λ = 0)
    ##############################
    if use_interp
        println("Solving VFI for baseline (flat tax, λ = 0) using interpolation...")
        V_baseline, policy_baseline = solve_vfi_interp(params_baseline, grid_a, grid_z, transition)
    else
        println("Solving VFI for baseline (flat tax, λ = 0) using grid search...")
        V_baseline, policy_baseline = solve_vfi(params_baseline, grid_a, grid_z, transition)
    end

    asset_vals_baseline, a_idx_baseline, z_idx_baseline = simulate_stationary_distribution(policy_baseline, grid_a, grid_z, transition)
    agg_assets_baseline = mean(asset_vals_baseline)
    println("Average assets (baseline) = ", agg_assets_baseline)
    gini_assets_baseline = gini(asset_vals_baseline)
    println("Gini coefficient for assets (baseline) = ", gini_assets_baseline)
    after_tax_income_baseline = [(1 - params_baseline.τ) * (w_eq * grid_z[z_idx]) for z_idx in z_idx_baseline]
    gini_income_baseline = gini(after_tax_income_baseline)
    println("Gini coefficient for after-tax labor income (baseline) = ", gini_income_baseline)

    # plotting Value and Policy Functions for a median productivity state.
    mid_z = div(n_z, 2)
    p1 = plot(grid_a, V_baseline[:, mid_z], title="Value Function (Baseline)",
                xlabel="Assets", ylabel="Value", label="V(a,z_mid)")
    savefig(p1, "value_function_baseline.png")

    p2 = plot(grid_a, policy_baseline[:, mid_z], title="Policy Function (Baseline)",
                xlabel="Assets", ylabel="Next-period Assets", label="Policy(a,z_mid)")
    savefig(p2, "policy_function_baseline.png")

    ##############################
    # Solve Household Problem: Progressive Tax (λ = 0.15)
    ##############################
    λ_progressive = 0.15
    τ_progressive = find_tau_progressive(params_baseline, λ_progressive, grid_a, grid_z, transition)  # new τ to keep government revenue constant (approx)
    println("Progressive tax regime: λ = ", λ_progressive, ", τ = ", τ_progressive)

    # using the calibrated beta for the progressive regime as well.
    println("Solving for equilibrium r and w in progressive tax regime...")
    r_progressive, w_progressive, K_progressive = find_equilibrium_prices(params_baseline, grid_a, grid_z, transition, α, A, δ)
    Y_progressive = A * K_progressive^α
    println("Computed r = ", r_progressive, ", w = ", w_progressive, ", K = ", K_progressive, ", Y = ", Y_progressive)

    params_progressive = Params(beta_calibrated, γ, τ_progressive, λ_progressive, r_progressive, w_progressive, ρ, σ)

    if use_interp
        println("Solving VFI for progressive tax regime (λ = 0.15) using interpolation...")
        V_progressive, policy_progressive = solve_vfi_interp(params_progressive, grid_a, grid_z, transition)
    else
        println("Solving VFI for progressive tax regime (λ = 0.15) using grid search...")
        V_progressive, policy_progressive = solve_vfi(params_progressive, grid_a, grid_z, transition)
    end

    asset_vals_progressive, a_idx_progressive, z_idx_progressive = simulate_stationary_distribution(policy_progressive, grid_a, grid_z, transition)
    agg_assets_progressive = mean(asset_vals_progressive)
    println("Average assets (progressive) = ", agg_assets_progressive)
    gini_assets_progressive = gini(asset_vals_progressive)
    println("Gini coefficient for assets (progressive) = ", gini_assets_progressive)
    after_tax_income_progressive = [(1 - params_progressive.τ) * (w_progressive * grid_z[z_idx])^(1 - params_progressive.λ) for z_idx in z_idx_progressive]
    gini_income_progressive = gini(after_tax_income_progressive)
    println("Gini coefficient for after-tax labor income (progressive) = ", gini_income_progressive)

    p3 = plot(grid_a, V_progressive[:, mid_z], title="Value Function (Progressive)",
                xlabel="Assets", ylabel="Value", label="V(a,z_mid)")
    savefig(p3, "value_function_progressive.png")

    p4 = plot(grid_a, policy_progressive[:, mid_z], title="Policy Function (Progressive)",
                xlabel="Assets", ylabel="Next-period Assets", label="Policy(a,z_mid)")
    savefig(p4, "policy_function_progressive.png")

    ##############################
    # Lorenz Curves for Assets and After-Tax Income
    ##############################
    pop_baseline, lorenz_assets_baseline = lorenz_curve(asset_vals_baseline)
    pop_progressive, lorenz_assets_progressive = lorenz_curve(asset_vals_progressive)

    p5 = plot(pop_baseline, lorenz_assets_baseline, label="Baseline",
                title="Lorenz Curve for Assets", xlabel="Cumulative Population",
                ylabel="Cumulative Share")
    plot!(p5, pop_progressive, lorenz_assets_progressive, label="Progressive")
    savefig(p5, "lorenz_assets.png")

    pop_income_baseline, lorenz_income_baseline = lorenz_curve(after_tax_income_baseline)
    pop_income_progressive, lorenz_income_progressive = lorenz_curve(after_tax_income_progressive)

    p6 = plot(pop_income_baseline, lorenz_income_baseline, label="Baseline",
                title="Lorenz Curve for After-Tax Income", xlabel="Cumulative Population",
                ylabel="Cumulative Share")
    plot!(p6, pop_income_progressive, lorenz_income_progressive, label="Progressive")
    savefig(p6, "lorenz_income.png")

    ##############################
    # Report Equilibrium Statistics
    ##############################
    println("\n=== Equilibrium Statistics ===")
    println("Baseline Regime (λ = 0):")
    println("  Interest rate (r): ", r_eq)
    println("  Wage rate (w): ", w_eq)
    println("  Tax parameter (τ): ", τ_baseline)
    println("  Capital-to-output ratio (K/Y): ", K / Y)
    println("  Gini (assets): ", gini_assets_baseline)
    println("  Gini (after-tax income): ", gini_income_baseline)

    println("\nProgressive Regime (λ = 0.15):")
    println("  Interest rate (r): ", r_progressive)
    println("  Wage rate (w): ", w_progressive)
    println("  Tax parameter (τ): ", τ_progressive)
    println("  Capital-to-output ratio (K/Y): ", K_progressive / Y_progressive)
    println("  Gini (assets): ", gini_assets_progressive)
    println("  Gini (after-tax income): ", gini_income_progressive)

    ##############################
    # Data Analysis on Simulation Output
    ##############################
    println("\n--- Data Analysis: Baseline Regime ---")
    analyze_simulation(asset_vals_baseline, after_tax_income_baseline, "baseline")

    println("\n--- Data Analysis: Progressive Regime ---")
    analyze_simulation(asset_vals_progressive, after_tax_income_progressive, "progressive")
end

main()
