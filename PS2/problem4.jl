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


using NLopt
using Plots


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

# Define the CES production function
function ces_production(x1, x2, α, σ)
    if σ == 1
        # Cobb-Douglas case
        return x1^α * x2^(1 - α)
    else
        # General CES case
        return (α * x1^((σ - 1) / σ) + (1 - α) * x2^((σ - 1) / σ))^(σ / (σ - 1))
    end
end

# Cost function to minimize
function cost_function(x, grad, α, σ, w1, w2, y)
    if length(grad) > 0
        # Gradient calculation can be added here if needed.
        return NaN
    end
    # Cost: w1 * x1 + w2 * x2
    return w1 * x[1] + w2 * x[2]
end

# Constraint function (production function equals target y)
function constraint(x, grad, α, σ, y)
    if length(grad) > 0
        # Gradient calculation can be added here if needed.
        return NaN
    end
    # f(x1, x2) - y = 0
    return ces_production(x[1], x[2], α, σ) - y
end

# Solve optimization problem
function optimize_cost(α, σ, w1, w2, y)
    opt = Opt(:LN_COBYLA, 2)  # Using COBYLA algorithm
    lower_bounds!(opt, [0.0, 0.0])  # Nonnegativity constraint
    min_objective!(opt, (x, grad) -> cost_function(x, grad, α, σ, w1, w2, y))  # Objective function
    equality_constraint!(opt, (x, grad) -> constraint(x, grad, α, σ, y), 1e-8)  # Constraint
    initial_x = [1.0, 1.0]  # Initial guess
    (min_cost, minimizer) = optimize(opt, initial_x)
    return (min_cost, minimizer)
end

# Test the function
α = 0.5
σ_values = [0.25, 1, 4]
w1 = 1.0
w2 = 1.0
y = 1.0

for σ in σ_values
    min_cost, minimizer = optimize_cost(α, σ, w1, w2, y)
    println("σ = $σ: Minimum cost = $min_cost, x1 = $(minimizer[1]), x2 = $(minimizer[2])")
end



# 4. Plot the cost function and the input demand functions (x1 and x2)
# for three different values of σ: σ = 0.25, 1, 4 as a function of w1 for
# w2 = 1 and y = 1. Set α to 0.5. Your answer should be a single plot
# with three panels (cost, x1, and x2) and three lines in each panel
# (each one for a different value of σ). You do not need to write a
# function for this part, it is enough to write a script that first calls
# the functions you wrote in the first part and then plots the results


# Generate cost and input demand plots
function plot_cost_and_demand(α, σ_values, w1_range, w2, y)
    costs = []
    x1_values = []
    x2_values = []

    for σ in σ_values
        cost = []
        x1_demand = []
        x2_demand = []
        for w1 in w1_range
            _, minimizer = optimize_cost(α, σ, w1, w2, y)
            push!(cost, w1 * minimizer[1] + w2 * minimizer[2])
            push!(x1_demand, minimizer[1])
            push!(x2_demand, minimizer[2])
        end
        push!(costs, cost)
        push!(x1_values, x1_demand)
        push!(x2_values, x2_demand)
    end

    # Plot
    p1 = plot(w1_range, costs, label=["σ = $(σ)" for σ in σ_values], xlabel="w1", ylabel="Cost", title="Cost Function")
    p2 = plot(w1_range, x1_values, label=["σ = $(σ)" for σ in σ_values], xlabel="w1", ylabel="x1", title="Input Demand x1")
    p3 = plot(w1_range, x2_values, label=["σ = $(σ)" for σ in σ_values], xlabel="w1", ylabel="x2", title="Input Demand x2")
    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
end

# Parameters for the plot
w1_range = 0.1:0.1:2.0
plot_cost_and_demand(α, σ_values, w1_range, w2, y)
