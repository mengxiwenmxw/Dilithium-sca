import time 
import numpy as np
import cupy as cp

# a = cp.random.rand(10, 10)
# b = cp.random.rand(10, 10)
# cp.dot(a, b)  # 触发编译

# # 生成数据
# numpy_vec = np.random.rand(1, 300000).astype(np.float32)
# numpy_mat = np.random.rand(300000, 1000).astype(np.float32)

# cupy_vec = cp.array(numpy_vec)  # 复制数据到 GPU
# cupy_mat = cp.array(numpy_mat)

# # NumPy 计算 (CPU)
# start = time.time()
# numpy_result = numpy_vec @ numpy_mat
# numpy_time = time.time() - start
# print(f"NumPy 耗时: {numpy_time:.5f} 秒")

# # CuPy 计算 (GPU)
# start = time.time()
# cupy_result = cupy_vec @ cupy_mat
# cp.cuda.Stream.null.synchronize()  # 等待 GPU 完成
# cupy_time = time.time() - start
# print(f"CuPy 耗时: {cupy_time:.5f} 秒")

# # 加速比
# print(f"\n加速比: NumPy / CuPy = {numpy_time / cupy_time:.1f}x")
print(f">>> Start to mk dir for key {0} in {0}:\n\
              -{0}\n\
                |--power_traces\n\
                |--averaged\n\
                    |--none\n\
                    |--align\n\
                    |--denoise\n\
                    |--align-denoise\n\
            ")