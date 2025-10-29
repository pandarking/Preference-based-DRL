import numpy as np

# 加载 NPZ 文件
data = np.load('file.npz')

# 查看文件中包含的所有数组名称
print("Arrays in the NPZ file:", data.files)

# # 读取特定数组
# array_name = 'example_array'  # 用你实际的数组名称替换
# array_data = data[array_name]
#
# # 处理读取到的数组数据
# # 这里假设数据是一个 NumPy 数组
# print("Shape of", array_name, ":", array_data.shape)
# print(array_data)

# 关闭 NPZ 文件
data.close()
