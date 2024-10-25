function odd_or_even(n)
    if n % 2 == 0
        println("Even")
    else
        println("Odd")
    end
end

odd_or_even(2)

function odd_or_even2(n)
    if iseven(n) == true
        println("Even")
    else
        println("Odd")
    end
end

odd_or_even2(7)