using LinearAlgebra
using Distributions

# Define the transition matrix P
P = [0.5 0.3 0.2;
     0.2 0.7 0.1;
     0.3 0.3 0.4]

# Define the states for Z and X
Z_states = [1, 2, 3]  # Corresponds to z1, z2, z3
X_states = 0:5        # Possible values of X_t

# Define the policy function σ(Xt, Zt)
function policy(Xt, Zt)
    if Zt == 1
        return 0  # Reset to 0 if Zt = z1
    elseif Zt == 2
        return Xt  # Stay the same if Zt = z2
    elseif Zt == 3
        return min(Xt + 1, 3)  # Increment if Xt ≤ 4, reset to 3 if Xt = 5
    end
end

# Compute the joint transition matrix for (Xt, Zt)
function joint_transition_matrix(P, X_states, Z_states)
    nX = length(X_states)
    nZ = length(Z_states)
    joint_P = zeros(nX * nZ, nX * nZ)

    for i in 1:nX, j in 1:nZ  # Loop over (Xt, Zt)
        current_index = (j - 1) * nX + i
        Xt = X_states[i]
        Zt = Z_states[j]

        for k in 1:nZ  # Transition probabilities for Zt -> Zt+1
            Zt_next = Z_states[k]
            prob_Z = P[j, k]
            Xt_next = policy(Xt, Zt_next)
            Xt_next_index = findfirst(==(Xt_next), X_states)
            next_index = (k - 1) * nX + Xt_next_index
            joint_P[current_index, next_index] += prob_Z
        end
    end
    return joint_P
end

# Calculate the stationary distribution
function stationary_distribution(joint_P)
    eigenvalues, eigenvectors = eigen(joint_P')
    stationary = eigenvectors[:, findfirst(≈(1.0), eigenvalues)]
    stationary /= sum(stationary)  # Normalize to sum to 1
    return stationary
end

# Compute the marginal distribution of Xt
function marginal_distribution(stationary, X_states, Z_states)
    nX = length(X_states)
    nZ = length(Z_states)
    marginal = zeros(nX)

    for i in 1:nX
        marginal[i] = sum(stationary[i:nX:end])
    end
    return marginal
end

# Compute the expected value of Xt
function expected_value_Xt(marginal, X_states)
    return sum(marginal .* X_states)
end

# Main computation
joint_P = joint_transition_matrix(P, X_states, Z_states)
stationary = stationary_distribution(joint_P)
marginal_X = marginal_distribution(stationary, X_states, Z_states)
expected_Xt = expected_value_Xt(marginal_X, X_states)

# Output results
println("Joint Transition Matrix:")
println(joint_P)
println("\nStationary Distribution (Joint):")
println(stationary)
println("\nMarginal Distribution of Xt:")
println(marginal_X)
println("\nExpected Value of Xt:")
println(expected_Xt)
