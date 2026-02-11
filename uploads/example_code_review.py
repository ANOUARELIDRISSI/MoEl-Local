### PROMPT ###
Please review the following Python code and provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance improvements
4. Security concerns
### END PROMPT ###

def calculate_average(numbers):
    sum = 0
    for num in numbers:
        sum = sum + num
    avg = sum / len(numbers)
    return avg

def find_max(data):
    max_val = data[0]
    for i in range(len(data)):
        if data[i] > max_val:
            max_val = data[i]
    return max_val

# Usage
nums = [1, 2, 3, 4, 5]
print("Average:", calculate_average(nums))
print("Maximum:", find_max(nums))
