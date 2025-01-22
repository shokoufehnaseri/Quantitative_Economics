using Plots

# Parameters
β = 0.95  # Discount factor (The discount factor reflects how much future utility is valued compared to present utility. A value of 0.95 indicates that future income or benefits are valued at 95% of their current value, which implies that workers are relatively patient and willing to wait for better job opportunities. This is a common assumption in economic models to balance short-term and long-term trade-offs.)
c = 1.0   # Unemployment benefit
wages = collect(10:1:100)  # Possible wages
π = fill(1 / length(wages), length(wages))  # Uniform wage distribution

function calculate_value_functions(p, β, c, wages, π)
    # Initialize value functions
    VU = zeros(length(wages))  # The initial value of VU is set to zero for all wage levels, reflecting no prior information about the utility of being unemployed. This provides a baseline for iterative updates.
    VE = zeros(length(wages))  # Similarly, VE starts at zero, allowing the model to iteratively build the utility of employment based on wages and other parameters.

    function update_values!(VU, VE, p, β, c, wages, π)
        for i in 1:length(wages)
            w = wages[i]
            VE[i] = w + β * ((1 - p) * VE[i] + p * sum(VU .* π))
            VU[i] = max(VE[i], c + β * sum(VU .* π))
        end
    end

    # Iterate until convergence
    for _ in 1:1000
        update_values!(VU, VE, p, β, c, wages, π)
    end

    return VU, VE
end

function calculate_reservation_wage(VU, VE, wages)
    # The reservation wage is determined by finding the lowest wage at which the value of being employed (VE) equals or exceeds the value of being unemployed (VU). This reflects the worker's threshold for accepting a job offer.
    for i in 1:length(wages)
        if VU[i] <= VE[i]
            return wages[i]
        end
    end
    return wages[end]
end

function calculate_acceptance_probability(wages, reservation_wage)
    return sum(w >= reservation_wage for w in wages) / length(wages)
end

function calculate_expected_duration(q)
    return 1 / q
end

# Varying p and computing results
ps = 0:0.1:1  # Range of separation probabilities
reservation_wages = Float64[]
acceptance_probabilities = Float64[]
unemployment_durations = Float64[]

for p in ps
    VU, VE = calculate_value_functions(p, β, c, wages, π)
    reservation_wage = calculate_reservation_wage(VU, VE, wages)
    push!(reservation_wages, reservation_wage)

    q = calculate_acceptance_probability(wages, reservation_wage)
    push!(acceptance_probabilities, q)

    unemployment_duration = calculate_expected_duration(q)
    push!(unemployment_durations, unemployment_duration)
end

# Plotting results
p1 = plot(ps, reservation_wages, label="Reservation Wage", xlabel="p", ylabel="Reservation Wage", title="Reservation Wage vs. p")
p2 = plot(ps, acceptance_probabilities, label="Acceptance Probability (q)", xlabel="p", ylabel="q", title="Acceptance Probability vs. p")
p3 = plot(ps, unemployment_durations, label="Expected Unemployment Duration", xlabel="p", ylabel="Duration", title="Unemployment Duration vs. p")

# Combine plots into a single layout
plot(p1, p2, p3, layout=(3, 1), size=(800, 1000))  # These plots collectively provide insights into how separation probability (p) influences key outcomes in the job search model. The first plot demonstrates how reservation wages decrease as p increases, reflecting workers' increased willingness to accept lower wages due to higher job insecurity. The second plot shows that acceptance probability (q) rises with p, indicating workers are more likely to accept job offers under high separation risk. Finally, the third plot illustrates that expected unemployment duration decreases with higher p, as workers spend less time unemployed due to increased job acceptance rates.
