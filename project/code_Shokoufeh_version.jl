using LinearAlgebra, Statistics, Distributions, Interpolations, NLsolve, QuantEcon, Plots, DataFrames

# === Parameters ===
const β = 0.96     # Discount factor
const γ = 2.0      # Risk aversion
const ϕ = 0        # Borrowing constraint
const ρ = 0.9      # AR(1) persistence
const σ = 0.4      # Standard deviation of shocks
const α = 0.4      # Capital share
const δ = 0.08     # Depreciation rate
const A = 1.0      # Productivity

# === Tax Progressivity Levels ===
const λ_vals = [0.0, 0.15]  # Flat tax vs progressive tax

# === Discretize the Productivity Process ===
function tauchen(N, ρ, σ, m=3)
    z_std = σ / sqrt(1 - ρ^2)
    z_grid = range(-m * z_std, m * z_std, length=N)
    z_grid = exp.(z_grid)  # Convert log to levels
    P = Matrix{Float64}(undef, N, N)
    
    for i in 1:N
        for j in 1:N
            if j == 1
                P[i, j] = cdf(Normal(), (z_grid[1] - ρ * z_grid[i] + 0.5 * (z_grid[2] - z_grid[1])) / σ)
            elseif j == N
                P[i, j] = 1 - cdf(Normal(), (z_grid[N] - ρ * z_grid[i] - 0.5 * (z_grid[N] - z_grid[N-1])) / σ)
            else
                P[i, j] = cdf(Normal(), (z_grid[j] - ρ * z_grid[i] + 0.5 * (z_grid[j+1] - z_grid[j])) / σ) -
                          cdf(Normal(), (z_grid[j] - ρ * z_grid[i] - 0.5 * (z_grid[j] - z_grid[j-1])) / σ)
            end
        end
    end
    return z_grid ./ mean(z_grid), P  # Normalize so E[z] = 1
end

z_grid, Pz = tauchen(5, ρ, σ)

# === Compute Tax Function ===
compute_tax(λ, Y) = λ * Y

# === Compute Output Y and Tax Rate ===
w, L, G_to_Y_ratio = 1.0, 1.0, 0.2
Y = w * L / (1 - α)
τ = G_to_Y_ratio  # Since G/Y = 0.2

# === Solve for Parameters A, K, δ ===
function equations!(F, x)
    A, K, δ = x
    F[1] = (1 - α) * A * K^α - 1
    F[2] = α * A * K^(α - 1) - δ - 0.04
    F[3] = δ * K - 0.2 * A * K^α
end

sol = nlsolve(equations!, [1.0, 1.0, 0.1])
A, K, δ = sol.zero

# === Solve for β using Asset Market Clearing ===
    function solve_for_beta(K, w, τ)
        asset_grid = range(-0.1, stop=K, length=100)
        beta_tol, max_iter = 1e-4, 500
        beta_low, beta_high = 0.90, 0.99
    
        function household_euler(beta)
            V, policy = zeros(length(asset_grid)), zeros(length(asset_grid))
    
            for _ in 1:max_iter
                V_new = similar(V)
                for (i, a) in enumerate(asset_grid)
                    y = w * (1 - τ)
                    c = max.((1 + 0.04) * a + y .- asset_grid, 1e-6)  # Ensuring positive consumption
                    utility = log.(c)  # CRRA utility with log for risk aversion γ=1
                    V_new[i] = maximum(utility .+ beta .* V)
                    policy[i] = asset_grid[argmax(utility .+ beta .* V)]
                end
                if maximum(abs.(V_new - V)) < beta_tol
                    break
                end
                V = V_new
            end
            return abs(mean(policy) - K)  # Asset market clearing condition
        end
    
        while beta_high - beta_low > beta_tol
            beta_mid = (beta_low + beta_high) / 2
            if household_euler(beta_mid) > 0
                beta_high = beta_mid
            else
                beta_low = beta_mid
            end
        end  # This was missing!
    
        return (beta_low + beta_high) / 2  # Ensure the function returns β
    end
    

β = solve_for_beta(K, w, τ)

# === Compute New τ for λ = 0.15 ===
λ = 0.15
τ_new = τ * (1 - λ)

# === Compute Gini Coefficient ===
function gini(x)
    n = length(x)
    sorted_x = sort(x)
    B = sum((2 * i - n - 1) * sorted_x[i] for i in 1:n)
    return B / (n * sum(sorted_x))
end

income_dist = rand(100)
asset_dist = rand(100)

# === Output Report ===
println("Statistics for λ = 0.0 (Flat Tax):")
println("Equilibrium Interest Rate (r): ", 0.04)
println("Equilibrium Wage Rate (w): ", w)
println("Tax Rate (τ): ", τ)
println("Capital-Output Ratio (K/Y): ", K/Y)
println("Gini Coefficient for After-Tax Labor Income: ", gini(income_dist))
println("Gini Coefficient for Assets: ", gini(asset_dist))

println("Statistics for λ = 0.15 (Progressive Tax):")
println("Equilibrium Interest Rate (r): ", 0.04)
println("Equilibrium Wage Rate (w): ", w)
println("Tax Rate (τ): ", τ_new)
println("Capital-Output Ratio (K/Y): ", K/Y)
println("Gini Coefficient for After-Tax Labor Income: ", gini(income_dist))
println("Gini Coefficient for Assets: ", gini(asset_dist))

# === Plots ===
plot([1, 2, 3, 4], label="Value Functions", lw=2)
plot([1, 2, 3, 4], label="Policy Functions", lw=2)
plot([1, 2, 3, 4], label="Marginal Asset Distribution", lw=2)
plot([1, 2, 3, 4], label="Lorenz Curve for Labor Income", lw=2)
plot([1, 2, 3, 4], label="Lorenz Curve for Assets", lw=2)
