
import numpy as np
from multiprocessing import Pool, shared_memory, Process, Queue
from tqdm import tqdm as tq
import multiprocessing as mp
import random
import queue
import os
import time
from collections import defaultdict
import json

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def incremental_pearson_corr(cumulative, new_power, new_h):
    """
    增量计算 Pearson 相关系数
    
    参数:
    cumulative - 累积统计量字典，包含:
        'n': 当前样本数量
        'sum_h': 中间值总和
        'sum_power': 功率轨迹总和 (向量)
        'sum_h_sq': 中间值平方和
        'sum_power_sq': 功率轨迹平方和 (向量)
        'sum_h_power': 中间值与功率轨迹乘积和 (向量)
        
    new_power - 新样本的功率轨迹 (向量)
    new_h - 新样本的中间值 (标量)
    
    返回:
    相关系数向量
    """
    n = cumulative['n'] + 1
    
    # 增量更新累积量
    delta_h = new_h - cumulative['sum_h'] / cumulative['n'] if cumulative['n'] > 0 else new_h
    cumulative['sum_h'] += new_h
    cumulative['sum_h_sq'] += new_h**2
    
    delta_power = new_power - cumulative['sum_power'] / cumulative['n'] if cumulative['n'] > 0 else new_power
    cumulative['sum_power'] += new_power
    cumulative['sum_power_sq'] += new_power**2
    
    # 更新协方差部分
    if cumulative['n'] > 0:
        cumulative['sum_h_power'] += delta_h * delta_power * cumulative['n'] / n
    
    cumulative['n'] = n
    
    # 计算均值和标准差
    mean_h = cumulative['sum_h'] / n
    mean_power = cumulative['sum_power'] / n
    
    var_h = cumulative['sum_h_sq'] / n - mean_h**2
    var_power = cumulative['sum_power_sq'] / n - mean_power**2
    
    # 防止分母为零
    var_h = np.maximum(var_h, 1e-10)
    var_power = np.maximum(var_power, 1e-10)
    
    # 计算协方差
    cov_h_power = cumulative['sum_h_power'] / n
    
    # 计算相关系数
    corr = cov_h_power / np.sqrt(var_h * var_power)
    
    # 处理无效值
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    
    return corr

# def incremental_pearson_corr(cumulative, new_power, new_h):
#     """
#     增量计算 Pearson 相关系数 - 使用Welford算法
#     """
#     n = cumulative['n'] + 1
    
#     # 保存旧的统计量
#     old_mean_h = cumulative['mean_h']
#     old_mean_power = cumulative['mean_power'].copy()
    
#     # 更新均值
#     cumulative['mean_h'] = old_mean_h + (new_h - old_mean_h) / n
#     cumulative['mean_power'] = old_mean_power + (new_power - old_mean_power) / n
    
#     # 更新协方差和方差
#     if n > 1:
#         # 更新协方差
#         cumulative['cov_sum'] += (new_h - old_mean_h) * (new_power - cumulative['mean_power'])
        
#         # 更新h的方差
#         cumulative['var_h_sum'] += (new_h - old_mean_h) * (new_h - cumulative['mean_h'])
        
#         # 更新power的方差
#         cumulative['var_power_sum'] += (new_power - old_mean_power) * (new_power - cumulative['mean_power'])
    
#     cumulative['n'] = n
    
#     # 当n<2时，相关系数未定义，返回0
#     if n < 2:
#         return np.zeros_like(new_power)
    
#     # 计算方差和协方差
#     var_h = cumulative['var_h_sum'] / (n - 1)
#     var_power = cumulative['var_power_sum'] / (n - 1)
#     cov_h_power = cumulative['cov_sum'] / (n - 1)
    
#     # 防止分母为零
#     denominator = np.sqrt(var_h * var_power)
#     denominator[denominator == 0] = np.inf
    
#     # 计算相关系数
#     corr = cov_h_power / denominator
    
#     # 处理无效值并限制范围
#     corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
#     corr = np.clip(corr, -1.0, 1.0)
    
#     return corr

def distance(plaintext, key):
    product = (key * 1729) % 3329
    temp = (plaintext + product) % 3329
    hwc = bin(temp).count('1')
    return hwc

def process_key(shared_mem_info, key, stop_trace_num):
    """
    处理单个密钥的所有轨迹数量
    """
    shm_name, shape, dtype = shared_mem_info
    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        power_trace_mat = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        # 初始化累积统计量
        cumulative = {
            'n': 0,
            'sum_h': 0,
            'sum_power': np.zeros(shape[1]),
            'sum_h_sq': 0,
            'sum_power_sq': np.zeros(shape[1]),
            'sum_h_power': np.zeros(shape[1])
        }
        
        correlations = []
        
        for trace_num in range(1, stop_trace_num + 1):
            # 直接使用明文作为行索引
            plaintext = trace_num - 1
            power_data = power_trace_mat[plaintext]
            h_val = distance(plaintext, key)
            
            # 增量计算相关系数
            corr = incremental_pearson_corr(cumulative, power_data, h_val)
            max_corr = np.max(np.abs(corr))
            correlations.append(max_corr)
            
            # 每处理100个轨迹数打印一次进度
            if trace_num % 100 == 0 and key % 100 == 0:
                print(f"Key {key} - Trace {trace_num}/{stop_trace_num} - Max Corr: {max_corr:.4f}")
        
        return (key, correlations)
    
    finally:
        if 'existing_shm' in locals():
            existing_shm.close()

class GetCpaTraceNum:
    def __init__(self, power_trace_file, sample_number=5000, plaintext_number=3329, key_number=3329,
                 process_number=None, low_sample=None, high_sample=None):
        self.power_trace_file = power_trace_file
        self.sample_number = sample_number
        self.key_number = key_number
        self.plaintext_number = plaintext_number
        self.process_number = process_number or max(1, mp.cpu_count() - 2)
        
        # 自动适配CPU核心数
        max_processes = min(32, os.cpu_count() * 2)
        self.process_number = min(self.process_number, max_processes)
        
        if low_sample is not None:
            self.low_sample = low_sample
        else:
            self.low_sample = 0
        
        if high_sample is not None:
            self.high_sample = high_sample
        else:
            self.high_sample = sample_number
        
        self.sample_number = self.high_sample - self.low_sample
        self.power_trace_mat = None

    def read_power(self):
        """高效读取功率轨迹数据"""
        print(f"📊 读取功率轨迹数据 (样本范围: {self.low_sample}-{self.high_sample})")
        
        # 初始化功率矩阵
        self.power_trace_mat = np.zeros((self.plaintext_number, self.sample_number), dtype=np.float32)
        
        # 使用缓冲区减少内存分配次数
        buffer_size = 1000
        buffer = []
        current_index = 0
        
        with tq(total=self.plaintext_number, desc="读取功率轨迹") as bar:
            with open(self.power_trace_file, 'r') as pf:
                for line in pf:
                    if not line.strip():
                        continue
                    
                    try:
                        parts = line.split(':', 1)
                        if len(parts) < 2:
                            continue
                            
                        plaintext_str, power_trace_str = parts
                        plaintext = int(plaintext_str)
                        
                        # 只处理在范围内的明文
                        if 0 <= plaintext < self.plaintext_number:
                            power_trace = np.fromstring(power_trace_str, sep=' ', dtype=np.float32)
                            
                            # 应用样本范围
                            if self.low_sample < self.high_sample:
                                power_trace = power_trace[self.low_sample:self.high_sample]
                            
                            buffer.append((plaintext, power_trace))
                            
                            # 缓冲区满时批量处理
                            if len(buffer) >= buffer_size:
                                for p, trace in buffer:
                                    self.power_trace_mat[p] = trace
                                buffer = []
                            
                            bar.update(1)
                            current_index += 1
                            
                            if current_index >= self.plaintext_number:
                                break
                    
                    except Exception as e:
                        print(f"解析错误: {line.strip()} - {str(e)}")
        
        # 处理剩余数据
        if buffer:
            for p, trace in buffer:
                self.power_trace_mat[p] = trace
        
        print(f"✅ 成功读取 {self.plaintext_number} 条功率轨迹")

    def correlation_trace_num(self, stop_plaintext_num=None, output_file=None):
        if stop_plaintext_num is None:
            stop_plaintext_num = min(self.plaintext_number, 3000)  # 限制最大轨迹数
        
        if output_file is None:
            raise ValueError('需要指定输出文件路径')
        
        print(f"⚙️ 开始分析轨迹数量和密钥相关性 (轨迹数: 1-{stop_plaintext_num})")
        print(f"🔑 密钥总数: {self.key_number} | 进程数: {self.process_number}")
        
        # 创建共享内存存放能量迹矩阵
        start_time = time.time()
        shm = shared_memory.SharedMemory(create=True, size=self.power_trace_mat.nbytes)
        shared_trace_mat = np.ndarray(
            self.power_trace_mat.shape, 
            dtype=self.power_trace_mat.dtype, 
            buffer=shm.buf
        )
        np.copyto(shared_trace_mat, self.power_trace_mat)
        print(f"🔗 共享内存创建完成 (耗时: {time.time()-start_time:.2f}s)")
        
        # 准备进程池
        shared_mem_info = (shm.name, self.power_trace_mat.shape, self.power_trace_mat.dtype)
        
        # 分批处理密钥以减少内存压力
        chunk_size = min(100, max(10, self.key_number // (self.process_number * 2)))
        key_ranges = []
        for i in range(0, self.key_number, chunk_size):
            end = min(i + chunk_size, self.key_number)
            key_ranges.append((i, end))
        
        print(f"📦 密钥分块: {len(key_ranges)} 块 | 每块大小: {chunk_size} 密钥")
        
        # 使用进程池处理密钥
        results_dir = "tmp_results"
        os.makedirs(results_dir, exist_ok=True)
        
        with Pool(processes=self.process_number) as pool:
            results = []
            futures = []
            
            # 提交任务
            for start_key, end_key in key_ranges:
                keys = list(range(start_key, end_key))
                future = pool.apply_async(
                    process_key_range, 
                    (shared_mem_info, keys, stop_plaintext_num, results_dir)
                )
                futures.append(future)
            
            # 等待所有任务完成
            with tq(total=len(futures), desc="处理密钥批次") as pbar:
                for future in futures:
                    future.get()  # 等待完成
                    pbar.update(1)
        
        # 合并部分结果
        print("🔗 合并部分结果...")
        merge_partial_results(results_dir, output_file, self.key_number)
        
        # 清理共享内存
        shm.close()
        shm.unlink()
        
        # 清理临时文件
        for f in os.listdir(results_dir):
            os.remove(os.path.join(results_dir, f))
        os.rmdir(results_dir)
        
        print(f"✅ 分析完成! 结果保存至: {output_file}")
        print(f"⏱️ 总耗时: {time.time()-start_time:.2f}秒")
    
    def show_traces(self,trace_file=None,highlight_keys=None,stop_plaintext_number=3329):
        if trace_file is None:
            raise ValueError('需要指定文件路径')
        # 初始化功率矩阵
        trace_result = np.zeros((self.key_number, stop_plaintext_number), dtype=np.float32)
        # 使用缓冲区减少内存分配次数
        buffer_size = 1000
        buffer = []
        current_index = 0
        with tq(total=self.key_number, desc="读取相关系数曲线") as bar:
            with open(trace_file, 'r') as pf:
                for line in pf:
                    if not line.strip():
                        continue
                    try:
                        parts = line.split(':', 1)
                        if len(parts) < 2:
                            continue
                        key_str, correlation_str = parts
                        key = int(key_str)
                        
                        # 只处理在范围内的明文
                        if 0 <= key < self.key_number:
                            correlations = np.fromstring(correlation_str, sep=',', dtype=np.float32)
                            # 应用样本范围
                            buffer.append((key, correlations))
                            # 缓冲区满时批量处理
                            if len(buffer) >= buffer_size:
                                for p, trace in buffer:
                                    trace_result[p] = trace
                                buffer = []
                            bar.update(1)
                            current_index += 1
                            
                            if current_index >= self.key_number:
                                break
                    
                    except Exception as e:
                        print(f"解析错误: {line.strip()} - {str(e)}")
        
        # 处理剩余数据
        if buffer:
            for p, trace in buffer:
                trace_result[p] = trace
        
        print(f"✅ 成功读取 {self.key_number} 条相关系数曲线")

        print("📊 准备可视化结果...")
        high_contrast_colors = [   
            '#FFD700',  # 金黄色
            '#FF6347',  # 番茄红
            '#FF8C00',  # 深橙色
            '#FF4500',  # 橙红色
            '#FF1493',  # 深粉色
            '#8B0000',  # 深红色
            '#FFA500',  # 橙色
            '#B22222',  # 砖红色
            '#800000',  # 栗色
            '#FF4500',  # 橙红色
        ]
        # 获取所有密钥的相关系数数据
        # 从第4个数据开始
        all_corrs = np.array([trace_result[key,3:] for key in range(self.key_number)])
        print('Data read finish')
        # 创建图形和坐标轴
        fig = plt.figure(figsize=(14, 8))
        # 否则只创建单个视图
        ax = plt.subplot(1, 1, 1)

        # 绘制所有密钥的相关系数曲线 (高性能方式)
        # 使用透明浅色绘制所有曲线
        x = np.arange(3,stop_plaintext_number)
        segments = np.array([np.column_stack([x, y]) for y in all_corrs])
        norm = plt.Normalize(0, len(all_corrs))
        lc = LineCollection(segments, cmap='Greys', norm=norm, alpha=0.1, linewidth=0.5)
        ax.add_collection(lc)

        # 设置坐标轴范围
        ax.set_xlim(3, stop_plaintext_number)
        ax.set_ylim(0, 0.5)  # 相关系数范围
        #ax.set_ylim(-0.5, 0.35)  # 相关系数范围

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6)

        # 添加标签
        ax.set_xlabel('Trace number')
        ax.set_ylabel('Correlation')
        # 创建高对比度颜色列表（避免蓝色）
        
        # 突出显示特定密钥
        if highlight_keys:
            print(f"highlight key: {highlight_keys}")
            #colors = plt.cm.tab10(np.linspace(0, 1, len(highlight_keys)))
            for i, key in enumerate(highlight_keys):
                corr = trace_result[key].flatten()
                label = f'key {key}'
                ax.plot(corr, color=high_contrast_colors[i%10], linewidth=1, alpha=0.9, label=label)
            # 添加图例
            ax.legend(loc='upper right')

        # 添加标题
        title = f'CPA result ({self.key_number} keys)'
        if highlight_keys:
            title += f'\nhighlight key(s): {", ".join(map(str, highlight_keys))}'
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()


def process_key_range(shared_mem_info, keys, stop_trace_num, results_dir):
    """处理一组密钥"""
    shm_name, shape, dtype = shared_mem_info
    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        power_trace_mat = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
        results = {}
        for key in keys:
            #初始化累积统计量
            cumulative = {
                'n': 0,
                'sum_h': 0,
                'sum_power': np.zeros(shape[1]),
                'sum_h_sq': 0,
                'sum_power_sq': np.zeros(shape[1]),
                'sum_h_power': np.zeros(shape[1])
            }

            # cumulative = {
            #         'n': 0,
            #         'mean_h': 0.0,
            #         'mean_power': np.zeros(shape),
            #         'cov_sum': np.zeros(shape),
            #         'var_h_sum': 0.0,
            #         'var_power_sum': np.zeros(shape)
            # }
            
            correlations = []
            
            for trace_num in range(1, stop_trace_num + 1):
                plaintext = trace_num - 1
                power_data = power_trace_mat[plaintext]
                h_val = distance(plaintext, key)
                
                # 增量计算相关系数
                corr = incremental_pearson_corr(cumulative, power_data, h_val)
                max_corr = np.max(np.abs(corr))
                correlations.append(max_corr)
            
            results[key] = correlations
        
        # 批量保存结果到临时文件
        output_file = os.path.join(results_dir, f"partial_{keys[0]}_{keys[-1]}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f)
            
        return True
    
    finally:
        if 'existing_shm' in locals():
            existing_shm.close()

def merge_partial_results(results_dir, output_file, total_keys):
    """合并部分结果文件"""
    all_results = {}
    processed_keys = set()
    
    for filename in os.listdir(results_dir):
        if filename.startswith('partial_'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                partial_results = json.load(f)
                for key_str, correlations in partial_results.items():
                    key = int(key_str)
                    if key not in processed_keys:
                        all_results[key] = correlations
                        processed_keys.add(key)
    
    # 检查是否所有密钥都已处理
    if len(all_results) < total_keys:
        print(f"⚠️ 警告: 只有 {len(all_results)}/{total_keys} 个密钥被处理")
    
    # 按密钥排序并写入最终结果
    with open(output_file, 'w') as f:
        for key in sorted(all_results.keys()):
            corr_str = ",".join(f"{c:.4f}" for c in all_results[key])
            f.write(f"{key}:{corr_str}\n")
    
    print(f"📝 写入最终结果: {len(all_results)} 个密钥")


if __name__ == "__main__":
    # power_file = 'data/1234/average/average_cd_loop_32.txt'
    # trace_file = 'result/correlation_trace_num_b1234.txt'
    # power_file = 'data/2773/average/average_loop_25.txt'
    # trace_file = 'result/correlation_trace_num_b2773.txt'
    # power_file = 'data/666/average/average_cd_loop_20.txt'
    # trace_file = 'result/correlation_trace_num_b1234.txt'
    # power_file = 'data/2619/average/average_cd_loop_5.txt'
    # trace_file = 'result/correlation_trace_num_b2773.txt'
    # power_file = 'data/1/average/average_cd_loop_5.txt'
    # trace_file = 'result/correlation_trace_num_b1.txt'
    power_file = 'data/2619/average/average_cd_loop_25.txt'
    #trace_file = 'result/correlation_trace_num_b2619_loop25.txt'
    trace_file = 'result/correlation_trace_num_b2773.txt'
    mode = 1 # 0 calculate correlation;1 show;
    cpa = GetCpaTraceNum(
        power_trace_file=power_file,
        sample_number=5000,
        plaintext_number=3329,
        key_number=3329,
        process_number=16,  # 减少进程数以避免资源竞争
        low_sample=4300,
        high_sample=5000
    )
    if mode ==0 :
        cpa.read_power()
        cpa.correlation_trace_num(
            stop_plaintext_num=3329,  # 减少轨迹数量以加快计算
            output_file=trace_file
        )
    elif mode == 1:
        cpa.show_traces(trace_file= trace_file,
        highlight_keys=[2773,556],
        # highlight_keys=[1234,2095],
        # highlight_keys=[2619,710],
        # highlight_keys=[666,2663],
        # highlight_keys=[2773,556],
        # highlight_keys=[1,3328],
        stop_plaintext_number=3329)