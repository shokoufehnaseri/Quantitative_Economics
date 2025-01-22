include("ngm.jl") 
using DataFrames, Plots

#Function to compute the convergence table
function convergence_table(γ_values, β, α, δ, k0, k_star)
    results = []
    for γ in γ_values
        model = NGMProblem(β=β, α=α, δ=δ, σ=γ, n=300)
        _, policy, _, _, _ = vfi(model)
        k_path = [k0]
        while true
            k_current = k_path[end]
            nearest_idx = argmin(abs.(model.k_grid .- k_current))
            k_next = policy[nearest_idx]
            push!(k_path, k_next)
            if abs(k_star - k_next) < 0.5 * abs(k_star - k0)
                push!(results, (γ, length(k_path)))
                break
            end
        end
    end
    return DataFrame(γ=first.(results), periods=last.(results))
end

#Function to generate the 4-panel plot
function plot_convergence(γ_values, β, α, δ, k0, time_horizon=100)
    capital_plot = plot(title="Capital", xlabel="Time", ylabel="Capital")
    output_plot = plot(title="Output", xlabel="Time", ylabel="Output")
    investment_plot = plot(title="Investment/Output", xlabel="Time", ylabel="Investment/Output")
    consumption_plot = plot(title="Consumption/Output", xlabel="Time", ylabel="Consumption/Output")

    for γ in γ_values
        model = NGMProblem(β=β, α=α, δ=δ, σ=γ, n=300)
        _, policy, _, _, _ = vfi(model)
        k_path = [k0]
        y_path = []
        i_path = []
        c_path = []

        for t in 1:time_horizon
            k_current = k_path[end]
            nearest_idx = argmin(abs.(model.k_grid .- k_current))
            y = model.f(k_current)
            k_next = policy[nearest_idx]
            i = k_next - (1 - δ) * k_current
            c = y - i

            push!(k_path, k_next)
            push!(y_path, y)
            push!(i_path, i / y)
            push!(c_path, c / y)
        end

        plot!(capital_plot, 1:time_horizon, k_path[1:time_horizon], label="γ=$γ")
        plot!(output_plot, 1:time_horizon, y_path, label="γ=$γ")
        plot!(investment_plot, 1:time_horizon, i_path, label="γ=$γ")
        plot!(consumption_plot, 1:time_horizon, c_path, label="γ=$γ")
    end

    return plot(capital_plot, output_plot, investment_plot, consumption_plot, layout=(2, 2), legend=:topleft)
end

# Parameters
β = 0.95
α = 0.3
δ = 0.05
k_star = ((β^(-1) - 1 + δ) / α)^(1 / (α - 1))
k0 = 0.5 * k_star
γ_values = [0.5, 1.0, 2.0]

# Gen Convergence Table
table = convergence_table(γ_values, β, α, δ, k0, k_star)
println("Convergence Table:")
println(table)

# Gen the 4 Panel Figure
fig = plot_convergence(γ_values, β, α, δ, k0)
display(fig)
