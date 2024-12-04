# Problem 4: Production
# A firm uses two inputs x1 and x2 to produce a single output y. The
# production function is given by
# f (x1, x2 ) =
# (
# αxσ−1σ
# 1 + (1 −α) x σ−1σ
# 2
# ) σσ−1
# ,
# where α ∈ (0, 1) and σ ≥ 0. This function is know as CES (constant
# elasticity of substitution) production function, and the parameter σ
# represents the elasticity of substitution. Notice σ = 1 corresponds to
# the Cobb-Douglas production function (i.e. f (x1, x2 ) = xα1 x1−α2 ).5 5 You can check it by taking the limit of
# the CES production function as σ →1.The firm minimizes its expenditure on inputs x1 and x2 subject to
# the production target y. The problem can be written as
# minx1 ,x2
# w1 x1 + w2 x2
# s.t. f (x1, x2 ) = y,
# where w1 and w2 are the prices of inputs x1 and x2, respectively. We
# also require that x1, x2 ≥ 0 - these are nonnegativity constraints on
# inputs.
# The value of w1 x1 + w2 x2 at the optimum is called the cost func-
# tion, C(w1, w2, y).
# Steps to follow:
# 1. Write a function that takes α, σ and x1 and x2 as arguments and
# creates a contour plot of the production function f (x1, x2 ) for
# x1 ≥0, x2 ≥0.

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve, Optim, Roots, Calculus

# CES production function
function ces_production_function(α, σ, x1, x2)
    if σ == 1
        return x1^α * x2^(1 - α)  
    else
        return (α * x1^((σ - 1) / σ) + (1 - α) * x2^((σ - 1) / σ))^(σ / (σ - 1))
    end
end

# Contour plot of the CES production function
function ces_contour_plot(α, σ, x_range, y_range)
    Z = [ces_production_function(α, σ, x, y) for x in x_range, y in y_range]
    contour(
        x_range,
        y_range,
        Z,
        xlabel="x1",
        ylabel="x2",
        title="σ = $σ",
        color=:viridis,
        lw=1,
        cbar=false
    )
end

# 2. Create the above plot for α = 0.5 and σ = 0.25, 1, 4. Your answer
# should be a single plot with three panels (each one for a different
# value of σ). You do not need to write a function for this part, it is
# enough to write a script that calls the function you wrote in the
# first part.


α = 0.5
x_range = 0.1:0.1:10
y_range = 0.1:0.1:10
σ_values = [0.25, 1, 4]

plots = [ces_contour_plot(α, σ, x_range, y_range) for σ in σ_values]
plot(plots..., layout=(1, 3), size=(1200, 400))





# 3. Write a function that takes α, σ, w1, w2, and y as arguments and
# returns the cost function and x1 and x2. Inside this function, you
# will have to solve a constrained minimization problem (because
# you have nonnegativity constraints). You can use whatever pack-
# age you want. Make sure that you test that optimization works
# problem set 2 5
# correctly and finds the correct solution. Note that the σ = 1 case
# might need a special treatment.

function cost_function(x, w1, w2)
    return w1 * x[1] + w2 * x[2]  # Simple cost calculation
end

function ces_function(α, σ, x)
    if σ == 1
        # Special case for Cobb-Douglas production function
        return x[1]^α * x[2]^(1 - α)
    else
        # General case for CES production function
        term1 = α * x[1]^((σ - 1) / σ)
        term2 = (1 - α) * x[2]^((σ - 1) / σ)
        return (term1 + term2)^(σ / (σ - 1))
    end
end

function ces_cost_function(α, σ, w1, w2, y)
    
    objective(x) = cost_function(x, w1, w2)
    

    constraint(x) = ces_function(α, σ, x) - y

    # our guess
    x0 = [1.0, 1.0] 
    
    result = optimize(
        x -> objective(x) + 1e6 * abs(constraint(x)), 
        [1e-6, 1e-6],  # Lower bounds for x1, x2 
        [Inf, Inf],    # Upper bounds for x1, x2 (no upper limit)
        x0,
        Fminbox(BFGS()),
        Optim.Options(store_trace=true, extended_trace=true, iterations=5000)
    )

    
    x_opt = Optim.minimizer(result)
    cost_opt = Optim.minimum(result)

    return cost_opt, x_opt
end

# Test the function
α = 0.5
σ = 0.5
w1 = 2.0
w2 = 1.5
y = 10.0



cost, x_opt = ces_cost_function(α, σ, w1, w2, y)
println("Optimal cost: $cost")
println("Optimal x1, x2: $x_opt")


# 4. Plot the cost function and the input demand functions (x1 and x2)
# for three different values of σ: σ = 0.25, 1, 4 as a function of w1 for
# w2 = 1 and y = 1. Set α to 0.5. Your answer should be a single plot
# with three panels (cost, x1, and x2) and three lines in each panel
# (each one for a different value of σ). You do not need to write a
# function for this part, it is enough to write a script that first calls
# the functions you wrote in the first part and then plots the results

using Plots

# Parameters
α = 0.5
w2 = 1.0
y = 1.0

σ_values = [0.25, 1.0, 4.0]  # Different values of σ
w1_range = 0.5:0.1:5.0       # Range of w1 values

# Initialize arrays to store results
costs = Dict(σ => [] for σ in σ_values)
x1_values = Dict(σ => [] for σ in σ_values)
x2_values = Dict(σ => [] for σ in σ_values)
# Compute results for each value of σ and w1
for σ in σ_values
    for w1 in w1_range
        cost, x_opt = ces_cost_function(α, σ, w1, w2, y)
        push!(costs[σ], cost)
        push!(x1_values[σ], x_opt[1])
        push!(x2_values[σ], x_opt[2])
    end
end


# Plotting
plot1 = plot(
    w1_range,
    [costs[σ] for σ in σ_values],
    label=[string("σ = ", σ) for σ in σ_values],
    xlabel="w1",
    ylabel="Cost",
    title="Cost Function",
    legend=:topright
)

plot2 = plot(
    w1_range,
    [x1_values[σ] for σ in σ_values],
    label=[string("σ = ", σ) for σ in σ_values],
    xlabel="w1",
    ylabel="x1",
    title="Input Demand: x1",
    legend=:topright
)

plot3 = plot(
    w1_range,
    [x2_values[σ] for σ in σ_values],
    label=[string("σ = ", σ) for σ in σ_values],
    xlabel="w1",
    ylabel="x2",
    title="Input Demand: x2",
    legend=:topright
)

# Combine the plots
plot(plot1, plot2, plot3, layout=(3, 1), size=(800, 1000))


