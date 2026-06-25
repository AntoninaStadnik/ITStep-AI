# Створіть масив:
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# 13 14 15 16
# Використовуючи індекси виведіть:
import numpy as np

# nums = np.arange(1,17)
# new_nums = nums.reshape(4,4)
# print(new_nums)
# print(new_nums.dtype)

# ● число 14
#print(nums[13])

# ● третій рядок
#print(new_nums[2])

# ● перший стовпчик
#print(new_nums[:, 1])

# ● верхню половину
# print(new_nums[0:2])

# ● замініть числа в рядках 2-3 на 100
# new_nums[1:3] = 100
# print(new_nums)

# ● зробіть другий рядок таким як останній рядок
# print(new_nums[1, :])
# new_nums[1, :] = new_nums[3, :]
# print(new_nums)

# У масиві з попереднього завдання створіть маску для
# парних чисел. З її допомогою
# mask = new_nums % 2 == 0
# print(mask)

# ● виведіть самі числа
# print(new_nums[mask])

# ● замініть їх на 100
# new_nums[mask] = 100
# print(new_nums)

# Створіть 2 масиви типу uint8:
# Масив 1: 128 200 10
# Масив 2: 250 10 34
# Об’єднайте їх у пропорції 20% першого масив + 80%
# другого масиву. В результаті має бути тип даних uint8 та
# числа в діапазоні 0-255

integers1 = np.array([128, 200, 10], dtype=np.uint8)
integers2 = np.array([250, 10, 34], dtype=np.uint8)

result = 0.2 * integers1 + 0.8 * integers2

result = result.astype(np.uint8)

print(result)