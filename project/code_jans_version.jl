using LinearAlgebra, Statistics, Distributions, Interpolations, NLsolve, QuantEcon
using Plots, DataFrames



# === Parameters ===
const β = 0.96     # Discount factor
const γ = 2.0      # Risk aversion
const ϕ = 0        # Borrowing constraint
const ρ = 0.9      # AR(1) persistence
const σ = 0.4      # Standard deviation of shocks
const α = 0.36     # Capital share (to be calibrated)
const δ = 0.08     # Depreciation rate
const A = 1.0      # Productivity

const λ_vals = [0.0, 0.15] # Tax progressivity levels
const grid_size = 100       # Asset grid points
const shock_size = 5        # Productivity grid points

# === Tauchen's Method for Productivity Process ===
function tauchen(N, μ, ρ, σ, m=3)
    z_std = σ / sqrt(1 - ρ^2)
    z_grid = range(-m * z_std, m * z_std, length=N)
    z_grid = exp.(z_grid) # Convert log to levels
    p = zeros(N, N)
    
    for i in 1:N
        for j in 1:N
            if j == 1
                p[i, j] = cdf(Normal(), (z_grid[1] - ρ * z_grid[i] + 0.5 * (z_grid[2] - z_grid[1])) / σ)
            elseif j == N
                p[i, j] = 1 - cdf(Normal(), (z_grid[N] - ρ * z_grid[i] - 0.5 * (z_grid[N] - z_grid[N-1])) / σ)
            else
                p[i, j] = cdf(Normal(), (z_grid[j] - ρ * z_grid[i] + 0.5 * (z_grid[j+1] - z_grid[j])) / σ) -
                          cdf(Normal(), (z_grid[j] - ρ * z_grid[i] - 0.5 * (z_grid[j] - z_grid[j-1])) / σ)
            end
        end
    end
    return z_grid, p
end

z_grid, Pz = tauchen(shock_size, 0, ρ, σ)
z_grid ./= mean(z_grid) # Normalize so E[z] = 1

# === Asset Grid ===
a_min, a_max = -ϕ, 50.0 # Borrowing constraint
agrid = range(a_min, a_max, length=grid_size)

# === Utility Function ===
function u(c)
    return c > 0 ? (c^(1-γ) - 1) / (1-γ) : -Inf
end

# === Compute Gini Coefficient ===
function gini_coefficient(x)
    x_sorted = sort(x)
    n = length(x)
    cumulative = cumsum(x_sorted)
    gini = 1 - (2 / n) * sum((n - i + 1) * x_sorted[i] for i in 1:n) / sum(x_sorted)
    return gini
end

# === Compute Lorenz Curve ===
function lorenz_curve(x)
    x_sorted = sort(x)
    cumulative = cumsum(x_sorted) / sum(x_sorted)
    return [0; cumulative]
end

# === Equilibrium Prices Solver ===
function T(y, τ, λ, ȳ)
    return y - (1 - τ) * (y / ȳ)^(1 - λ) * ȳ
end

function find_tau(λ, target_G, tol=1e-4)
    function revenue_error(τ)
        w, r, _, _, _ = equilibrium_prices(λ, τ[1], tol)
        G = sum(T(y, τ[1], λ, mean(z_grid * w)) for y in z_grid * w) / length(z_grid)
        println("Trying τ = ", τ[1], " → Revenue difference: ", G - target_G)  # Debug
        return G - target_G
    end

    res = nlsolve(x -> [revenue_error(x)], [0.2])

    if !res.f_converged
        println("WARNING: Could not find a valid τ for λ = $λ!")
        return 0.2  # Fallback to default tax rate
    end

    return res.zero[1]
end


function equilibrium_prices(λ, τ_guess=0.2, tol=1e-4, max_iter=500)
    global τ_fixed

    if λ == 0
        τ_fixed = τ_guess
    else
        # Solve for λ = 0 first to get target_G
        w, r, _, _, _ = equilibrium_prices(0, τ_guess, tol)
        ȳ = mean(z_grid * w)
        target_G = sum(T(y, τ_guess, 0, ȳ) for y in z_grid * w) / length(z_grid)

        # Solve for τ using a new function (no recursion)
        τ_fixed = find_tau_no_recursion(λ, target_G, tol)
    end

    # Initial guesses
    r = 0.04
    w = 1.0
    V = zeros(shock_size, grid_size)
    policy = zeros(shock_size, grid_size)

    for iter in 1:max_iter
        V_new, policy = solve_bellman(V, agrid, z_grid, w, r, λ, τ_fixed)

        max_change = maximum(abs.(V_new - V))
        if iter % 50 == 0
            println("Iteration $iter: Max change = ", max_change)
        end

        if max_change < tol
            println("✅ Converged in $iter iterations for λ = $λ with τ = $τ_fixed")
            break
        end
        
        if iter == max_iter
            println("❌ WARNING: Did NOT converge! Max change was $max_change")
        end

        V .= V_new
    end

    return w, r, τ_fixed, V, policy
end





# === Bellman Equation Solver ===
function solve_bellman(V, agrid, z_grid, w, r, λ, τ)
    V_new = similar(V)
    policy = similar(V)
    
    for iz in 1:shock_size
        for ia in 1:grid_size
            a = agrid[ia]
            z = z_grid[iz]
            y = z * w
            y_post_tax = (1 - τ) * y^(1-λ) # After-tax income
            
            max_val = -Inf
            best_a_next = a_min
            
            for ja in 1:grid_size
                a_next = agrid[ja]
                c = y_post_tax + (1 + r) * a - a_next
                
                if c > 0
                    expected_value = sum(Pz[iz, j] * V[j, ja] for j in 1:shock_size)
                    value = u(c) + β * expected_value
                    
                    if value > max_val
                        max_val = value
                        best_a_next = a_next
                    end
                end
            end
            V_new[iz, ia] = max_val
            policy[iz, ia] = best_a_next
        end
    end
    return V_new, policy
end

function find_tau_no_recursion(λ, target_G, tol=1e-4)
    function revenue_error(τ)
        # Compute revenue without solving full equilibrium
        w = 1.0
        r = 0.04
        ȳ = mean(z_grid * w)
        G = sum(T(y, τ[1], λ, ȳ) for y in z_grid * w) / length(z_grid)

        println("Trying τ = ", τ[1], " → Revenue difference: ", G - target_G)
        return G - target_G
    end

    res = nlsolve(x -> [revenue_error(x)], [0.2])

    if !res.f_converged
        println("❌ WARNING: Tax solver did NOT converge for λ = $λ!")
        return 0.2  # Default fallback
    end

    println("✅ Found τ = ", res.zero[1])
    return res.zero[1]
end


# === Solve for Two Equilibria ===
equilibria = Dict()
for λ in λ_vals
    equilibria[λ] = equilibrium_prices(λ)
end

# === Compute and Plot Results ===
for (λ, (w, r, τ, V, policy)) in equilibria
    println("Equilibrium for λ = $λ: w = $w, r = $r, τ = $τ")
    
    # Compute Gini coefficients
    asset_distribution = agrid
    gini_assets = gini_coefficient(asset_distribution)
    println("Gini coefficient for assets: ", gini_assets)
    
    income_distribution = [(1 - τ) * (z * w)^(1-λ) for z in z_grid]
    gini_income = gini_coefficient(income_distribution)
    println("Gini coefficient for after-tax labor income: ", gini_income)
    
    # Plot Lorenz Curves
    display(plot(lorenz_curve(asset_distribution), label="Assets", title="Lorenz Curve for λ = $λ"))
    display(plot!(lorenz_curve(income_distribution), label="After-tax Income"))
    
    # Plot Policy Function
    plot(agrid, policy[1, :], label="Low productivity", title="Policy Function for λ = $λ")
    plot!(agrid, policy[end, :], label="High productivity")
end
