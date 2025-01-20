using Statistics

X = 50.0            # utility from obtaining the orchid
C = 0.5             # cost per vendor approached
f = 1.0             # mental cost of approaching a vendor
q = 0.15            # prob a vendor has the orchid
p_min = 10.0        # min price
p_max = 100.0       # max price
prices = p_min:0.1:p_max  # discretized price range
num_vendors = 50    # total vendors at the festival

function v_T(n)
    return -C * n
end

function v_B(n, p)
    return X - p - C * n
end

function v_A(n, v_next, prices)
    expected_price_value = mean(max(v_B(n + 1, p), v_next) for p in prices)
    return -f + q * expected_price_value + (1 - q) * v_next
end


function solve_bellman()
    v = zeros(num_vendors + 1) 
    σ_approach = zeros(Int, num_vendors + 1) 
    σ_buy = zeros(Int, num_vendors + 1, length(prices)) 

    for n in num_vendors:-1:0
        v_terminate = v_T(n)

        if n < num_vendors
            v_next = v[n + 1]
            v_approach = v_A(n, v_next, prices)

            if v_approach > v_terminate
                v[n + 1] = v_approach
                σ_approach[n + 1] = 1
            else
                v[n + 1] = v_terminate
                σ_approach[n + 1] = 0
            end

            for (i, p) in enumerate(prices)
                if v_B(n + 1, p) > v_next
                    σ_buy[n + 1, i] = 1
                else
                    σ_buy[n + 1, i] = 0
                end
            end
        else
            v[n + 1] = v_terminate
        end
    end

    return v, σ_approach, σ_buy
end

value_function, policy_approach, policy_buy = solve_bellman()

#a
function compute_prob_buy(policy_buy)
    total_opportunities = num_vendors * length(prices)
    buy_decisions = sum(policy_buy)
    return buy_decisions / total_opportunities
end

#b
function compute_expected_price(policy_buy, prices)
    total_weighted_price = sum(p * policy_buy[n, i] for n in 1:num_vendors for (i, p) in enumerate(prices))
    total_buy_decisions = sum(policy_buy)
    return total_weighted_price / total_buy_decisions
end

#c
function compute_expected_vendors(policy_approach)
    return sum(policy_approach)
end

#d
function analyze_willingness_to_pay(policy_buy, prices)
    willingness = [
        mean(prices[findall(x -> x == 1, policy_buy[n, :])])
        for n in 1:num_vendors if any(policy_buy[n, :] .== 1)
    ]
    return willingness
end

prob_buy = compute_prob_buy(policy_buy)
expected_price = compute_expected_price(policy_buy, prices)
expected_vendors = compute_expected_vendors(policy_approach)
willingness_to_pay = analyze_willingness_to_pay(policy_buy, prices)

println("Results:")
println("a) Probability Basil buys the orchid: $prob_buy")
println("b) Expected price Basil pays (conditional on buying): $expected_price")
println("c) Expected number of vendors Basil will approach: $expected_vendors")
println("d) Willingness to pay higher prices over time:")
println(willingness_to_pay)
