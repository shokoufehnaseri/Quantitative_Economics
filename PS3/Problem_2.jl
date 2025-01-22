using Plots

β = 0.95  # Discount factor 
c = 1.0   # Unemployment benefit
wages = collect(10:1:100)  # Possible wages
π = fill(1 / length(wages), length(wages))  # Uniform wage distribution

function calculate_value_functions(p, β, c, wages, π)
    
    VU = zeros(length(wages))  
    VE = zeros(length(wages))  

    function update_values!(VU, VE, p, β, c, wages, π)
        for i in 1:length(wages)
            w = wages[i]
            VE[i] = w + β * ((1 - p) * VE[i] + p * sum(VU .* π))
            VU[i] = max(VE[i], c + β * sum(VU .* π))
        end
    end

    
    for _ in 1:1000
        update_values!(VU, VE, p, β, c, wages, π)
    end

    return VU, VE
end

function calculate_reservation_wage(VU, VE, wages)
    
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


ps = 0:0.1:1  
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

p1 = plot(ps, reservation_wages, label="Reservation Wage", xlabel="p", ylabel="Reservation Wage", title="Reservation Wage vs. p")
p2 = plot(ps, acceptance_probabilities, label="Acceptance Probability (q)", xlabel="p", ylabel="q", title="Acceptance Probability vs. p")
p3 = plot(ps, unemployment_durations, label="Expected Unemployment Duration", xlabel="p", ylabel="Duration", title="Unemployment Duration vs. p")


plot(p1, p2, p3, layout=(3, 1), size=(800, 1000))  # These plots collectively provide insights into how separation probability (p) influences key outcomes in the job search model. The first plot demonstrates how reservation wages decrease as p increases, reflecting workers' increased willingness to accept lower wages due to higher job insecurity. The second plot shows that acceptance probability (q) rises with p, indicating workers are more likely to accept job offers under high separation risk. Finally, the third plot illustrates that expected unemployment duration decreases with higher p, as workers spend less time unemployed due to increased job acceptance rates.
