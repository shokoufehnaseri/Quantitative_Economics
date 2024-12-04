function iterative_solver(f, x0, α; ϵ=1e-6, maxiter=1000)
    """
    Iterative solver for nonlinear equations using a dampened iterative method.

    Arguments:
    - f: The function to find the root for.
    - x0: Initial guess for the root.
    - α: Dampening parameter (0 ≤ α ≤ 1).
    - ϵ: Convergence tolerance (default 1e-6).
    - maxiter: Maximum number of iterations (default 1000).

    Returns:
    - flag: 0 if the solution is found before maxiter, 1 otherwise.
    - solution: Approximate root (or NaN if not found).
    - value: Value of f at the solution (or NaN if not found).
    - abs_diff: Absolute difference between x and g(x) for the final iteration.
    - x_values: Vector of all x values during iterations.
    - residuals: Vector of residuals during iterations.
    """
    x_values = Float64[x0]  
    residuals = Float64[]
    
    for i in 1:maxiter
        g_x = f(x0) + x0  
        x_next = (1 - α) * g_x + α * x0  
        residual = abs(x_next - x0)
        push!(residuals, residual)
        push!(x_values, x_next)

        if residual / (1 + abs(x0)) < ϵ
            return (0, x_next, f(x_next), abs(x_next - g_x), x_values, residuals)
        end

        x0 = x_next
    end

    return (1, NaN, NaN, NaN, x_values, residuals)
end

f(x) = (x + 1)^(1/3) - x

x0 = 1  
α = 0.5  
flag, solution, value, abs_diff, x_values, residuals = iterative_solver(f, x0, α)

println("Flag: $flag")
println("Solution: $solution")
println("Value at solution: $value")
println("Absolute difference: $abs_diff")
println("Last 5 x values: ", x_values[end-4:end])
println("Last 5 residuals: ", residuals[end-4:end])