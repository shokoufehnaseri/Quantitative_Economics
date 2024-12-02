using LinearAlgebra, Printf

#Exact solution function
function exact_solution(α, β)
    # Solve the system manually starting from the last equation
    x5 = 1
    x4 = x5 + 1
    x3 = x4 + 1
    x2 = x3 + 1
    x1 = α + (α - β) * x2 + β * x5
    return [x1, x2, x3, x4, x5]
end

#Function for numerical solution and metrics
function numerical_solution(α, β)
    # Define matrix A and vector b
    A = [
        1  -1   0  α - β  β;
        0   1  -1    0    0;
        0   0   1   -1    0;
        0   0   0    1   -1;
        0   0   0    0    1
    ]
    b = [α, 0, 0, 0, 1]

    # Compute exact solution
    x_exact = exact_solution(α, β)

    # Solve numerically
    x_numerical = A \ b

    # Calculate condition number and relative residual
    cond_number = cond(A)
    rel_residual = norm(A * x_numerical - b) / norm(b)

    return x_exact, x_numerical, cond_number, rel_residual
end

#Create table for different values of β
function create_table(α)
    β_values = [10^i for i in 0:12]  # β = 1, 10, ..., 10^12
    results = []

    for β in β_values
        x_exact, x_numerical, cond_number, rel_residual = numerical_solution(α, β)
        # Collect results for x1 (exact and numerical), condition number, and residual
        push!(results, (β, x_exact[1], x_numerical[1], cond_number, rel_residual))
    end

    return results
end

#Generate results for α = 0.1 and display the table
function display_table(α)
    results = create_table(α)
    println(@sprintf("%-15s %-15s %-15s %-15s %-15s", "β", "Exact x1", "Numerical x1", "Cond. Number", "Rel. Residual"))
    println("-"^75)
    for (β, x1_exact, x1_num, cond_num, rel_res) in results
        println(@sprintf("%-15e %-15.8f %-15.8f %-15.8f %-15.8e", β, x1_exact, x1_num, cond_num, rel_res))
    end
end

#Run the table display for α = 0.1
α = 0.1
display_table(α)
