# Julia translation of the Tax Reform Model

using Distributions, Plots, StatsBase

# Parameters
beta = 0.96  # Discount factor
gamma = 2.0  # Risk aversion
r = 0.04     # Interest rate
w = 1.0      # Wage rate
tau = 0.2    # Average tax rate

# Asset grid
amin = 0
amax = 50
agrid_size = 500
a_grid = range(amin, stop=amax, length=agrid_size)

# Productivity states and transition matrix
z_vals = [0.5, 0.8, 1.0, 1.2, 1.5]
z_probs = fill(1/length(z_vals), length(z_vals))

# Utility function
utility(c) = c > 0 ? (c^(1 - gamma)) / (1 - gamma) : -Inf

# Tax function
tax(y, tau, lambda_tax, y_avg) = y - (1 - tau) * (y / y_avg)^(1 - lambda_tax) * y_avg

# Value Function Iteration
function solve_model(beta, gamma, r, w, tau, lambda_tax, a_grid, z_vals, z_probs)
    V = zeros(length(z_vals), length(a_grid))
    policy_a = similar(V)
    policy_c = similar(V)
    y_avg = w * mean(z_vals)
    
    max_iter = 1000
    tol = 1e-6
    for it in 1:max_iter
        V_new = similar(V)
        for (i, z) in enumerate(z_vals)
            y = w * z
            T = tax(y, tau, lambda_tax, y_avg)
            y_post_tax = y - T
            for (j, a) in enumerate(a_grid)
                c = y_post_tax .+ (1 + r) * a .- a_grid
                utility_c = utility.(c)
                EV = dot(z_probs, V)
                V_choice = utility_c .+ beta * EV
                V_new[i, j] = maximum(V_choice)
                policy_a[i, j] = a_grid[argmax(V_choice)]
                policy_c[i, j] = c[argmax(V_choice)]
            end
        end
        if maximum(abs.(V_new - V)) < tol
            break
        end
        V = V_new
    end
    return V, policy_a, policy_c
end

# Simulate the stationary distribution
function simulate_stationary_distribution(policy_a, z_probs, a_grid, z_vals, sim_length=100000)
    a_idx = zeros(Int, sim_length)
    z_idx = rand(Categorical(z_probs), sim_length)
    
    for t in 2:sim_length
        a_next = policy_a[z_idx[t-1], a_idx[t-1]+1]
        a_idx[t] = searchsortedfirst(a_grid, a_next) - 1
    end
    
    a_sim = a_grid[a_idx .+ 1]
    z_sim = z_vals[z_idx]
    return a_sim, z_sim
end

# Calculate Gini coefficient
gini_coefficient(x) = gini(x)

# Calculate and plot Lorenz curves
function lorenz_curve(x)
    sorted_x = sort(x)
    x_cum = cumsum(sorted_x) / sum(sorted_x)
    x_cum = vcat(0.0, x_cum)
    return range(0.0, stop=1.0, length=length(x_cum)), x_cum
end

# Compare two economies
lambda_values = [0.0, 0.15]
results = Dict()

for lambda_tax in lambda_values
    V, policy_a, policy_c = solve_model(beta, gamma, r, w, tau, lambda_tax, a_grid, z_vals, z_probs)
    a_sim, z_sim = simulate_stationary_distribution(policy_a, z_probs, a_grid, z_vals)
    
    gini_assets = gini_coefficient(a_sim)
    after_tax_income = (w .* z_sim) .- tax.(w .* z_sim, tau, lambda_tax, w * mean(z_vals))
    gini_income = gini_coefficient(after_tax_income)
    
    K = mean(a_sim)
    L = mean(z_sim)
    Y = K^0.36 * L^0.64
    K_over_Y = K / Y
    
    results[lambda_tax] = Dict(
        "V" => V,
        "policy_a" => policy_a,
        "policy_c" => policy_c,
        "a_sim" => a_sim,
        "gini_assets" => gini_assets,
        "gini_income" => gini_income,
        "K_over_Y" => K_over_Y
    )

    # Plot Value Functions
    plot()
    for (i, z) in enumerate(z_vals)
        plot!(a_grid, V[i, :], label="z = $z")
    end
    xlabel!("Assets")
    ylabel!("Value Function")
    title!("Value Function (λ = $lambda_tax)")
    display(plot())

    # Plot Policy Functions (Assets)
    plot()
    for (i, z) in enumerate(z_vals)
        plot!(a_grid, policy_a[i, :], label="z = $z")
    end
    plot!(a_grid, a_grid, linestyle=:dash, label="45-degree line")
    xlabel!("Assets")
    ylabel!("Next Period Assets")
    title!("Policy Function (λ = $lambda_tax)")
    display(plot())

    # Plot Asset Distribution
    histogram(a_sim, bins=50, normalize=true, label=false)
    xlabel!("Assets")
    ylabel!("Density")
    title!("Stationary Distribution of Assets (λ = $lambda_tax)")
    display(plot())

    # Lorenz Curves
    x_lorenz_assets, y_lorenz_assets = lorenz_curve(a_sim)
    x_lorenz_income, y_lorenz_income = lorenz_curve(after_tax_income)
    
    plot(x_lorenz_assets, y_lorenz_assets, label="Assets")
    plot!(x_lorenz_income, y_lorenz_income, label="After-tax Income")
    plot!([0,1], [0,1], linestyle=:dash, label="Perfect Equality")
    xlabel!("Cumulative Share of Population")
    ylabel!("Cumulative Share")
    title!("Lorenz Curves (λ = $lambda_tax)")
    display(plot())
end

# Report statistics
for (lambda_tax, res) in results
    println("\nEconomy with λ = $lambda_tax:")
    println("Equilibrium Interest Rate (r): $r")
    println("Equilibrium Wage Rate (w): $w")
    println("Tax Rate (τ): $tau")
    println("Capital-to-Output Ratio (K/Y): $(res["K_over_Y"])\n")
    println("Gini Coefficient (Assets): $(res["gini_assets"])\n")
    println("Gini Coefficient (After-tax Income): $(res["gini_income"])\n")
end
