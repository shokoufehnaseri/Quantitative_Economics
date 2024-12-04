function NPV(r, C)
    T = length(C)
    return sum(C[t] / (1 + r)^t for t in 1:T)
end

using Roots, LinearAlgebra, NLsolve, Optim

function internal_rate(C)
    a, b = -0.9, 2.0
    wrapped_NPV(r) = NPV(r, C)

    if wrapped_NPV(a) * wrapped_NPV(b) >= 0
        return "bad interval"
    end

    output = find_zero(wrapped_NPV, (a, b))
    return output

end

println(internal_rate([-5, 0, 0, 2.5, 5]))