using Plots
#Problem3
function my_factorial(n)
    result = 1
    for i in (1:n)
        result = result * i
    end
    return result
end

#Problem4
function count_positives(arr)
    counter = 0
    for i in (1:length(arr))
        if arr[i] > 0 
            counter = counter + 1
        else 
            counter = counter
        end
    end
    return counter
end

#Problem5
