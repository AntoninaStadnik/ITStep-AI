# Створіть масив з числами від 1 до 10. Виведіть його, його
# розмір, тип даних.
# Змініть розмір масиву на (5, 2). Знову виведіть масив,
# розмір та тип даних

import numpy as np

# nums = np.arange(1,11)
# print(nums)
# print(nums.shape)
# print(nums.dtype)
#
# new_nums = nums.reshape(2,5)
# print(new_nums)
# print(new_nums.shape)
# print(new_nums.dtype)

# Створіть масив:
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# Використовуючи індекси виведіть:
nums1 = np.arange(1,13)
new_nums1 = nums1.reshape(3,4)
print(new_nums1)
print(new_nums1.shape)
print(new_nums1.dtype)

# # ● число 7
# print(new_nums1[1,2])
#
# # ● другий рядок
# print(new_nums1[1,:])
#
# # ● останній стовпчик
# print(new_nums1[:, 3])
#
# # ● праву половину
# print(new_nums1[:, 2:4])
#
# # ● жовту область
# print(new_nums1[1:3, 1:3])
#
# # ● замініть жовту область на -1
# new_nums1[1:3, 1:3] = -1
# print(new_nums1)
#
# # ● зробіть перший стовпчик таким самим як і другий
# print(new_nums1[:, 0])
# new_nums1[:, 0] = new_nums1[:, 1]
# print(new_nums1)

# У масиві з попереднього завдання створіть маску для
# чисел які більші за 6. З її допомогою

# ● виведіть кількість чисел більших за 6
mask = new_nums1 > 6
print(mask.sum())

# ● виведіть самі числа
print(new_nums1[mask])

# ● до кожного числа яке відповідає масці додайте 10
new_nums1[mask] += 10
print(new_nums1)

# ● кожне число що не відповідає масці помножте на -1
new_mask = ~mask
print(new_mask)

new_nums1[new_mask] *= -1
print(new_nums1)

# ● замініть ці числа які відповідають масці на відповідні
# їм з масиву
# 1 0 1 0
# 0 1 0 1
# 1 0 1 0

nums2 = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
])

new_nums1[new_mask] = nums2[new_mask]
print(new_nums1)


# Створіть масив
# -10 24 35
# 250 -6 7
# 12 180 11
# -2 -45 -26
# Усі числа менші за 0 замініть на 0.
# Усі числа більші за 100 замініть на 100

# integers = np.array([
#     [-10, 24, 35],
#     [250, -6, 7],
#     [12, 180, 11],
#     [-2 -45 -26]
# ])
#
# Створіть масив типу uint8
# 10 4 25 40 200
# |Помножте всі значення на 2. Результат має бути типу
# uint8 а всі значення в діапазоні 0-255
# Помножте всі значення на 1.5. Результат має бути типу
# uint8 а всі значення в діапазоні 0-255

integers = np.array([10, 4, 25, 40, 200])

integers = integers.astype(np.uint8)
integers = integers.astype(np.int64)

integers *= 2
mask = integers > 255
integers[mask] = 255
print(integers)

print(mask)
print(integers)
print(integers.dtype)
print(integers.shape)

integers = integers.astype(np.uint8)
print(integers)
print(integers.dtype)
print(integers.shape)



integers = integers * 1.5
mask = integers > 255
integers[mask] = 255

integers = integers.astype(np.uint8)

print(mask)
print(integers)
print(integers.dtype)
print(integers.shape)