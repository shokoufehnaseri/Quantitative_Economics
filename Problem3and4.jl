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
function count_positives_broadcasting(arr)
    
    positive_elements = arr .> 0
    sum(positive_elements)
end

# problem 6
function standard_deviation(x)
    n = length(x)
    # Step 1: Calculate the mean
    μ = sum(x) / n

    # Step 2: Calculate the squared differences from the mean
    squared_d = (x .- μ) .^ 2

    # Step 3: Calculate the variance
    variance = sum(squared_d) / (n - 1)

    # Step 4: Calculate the standard deviation
    SD = sqrt(variance)

    # Step 5: Return the SD value
    return SD
end


