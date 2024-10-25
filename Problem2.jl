function compare_three(a, b, c)
    if a > 0 && b > 0 && c > 0
        println("All numbers are positive")
    elseif a == 0 && b == 0 && c == 0
        println("All numbers are zero")
    else
        println("At least one number is not positive")
    end
end

compare_three(-1, 2, 6)
compare_three(-1, 2, -3)
compare_three(1, 2, 0)