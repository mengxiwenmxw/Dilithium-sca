
import numpy as np
from multiprocessing import Pool,shared_memory
from tqdm import tqdm as tq
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import json
import random

def generate_unique_random_numbers(n,n_max=3329):

    if n < 0 or n >= n_max:
        raise ValueError(f"n必须在0到{n_max-1}之间")
    # 生成0到3328的整数序列
    population = list(range(0, 3329))
    
    # 随机抽取n个不重复的数字
    return random.sample(population, n)

def calculate_correlation(x,y):
    """
    pearson correlation
    :param x:
    :param y:
    :return: r
    """
    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 计算分子
    numerator = np.sum((x - mean_x) * (y - mean_y))

    # 计算分母
    denominator = np.sqrt(np.sum((x - mean_x) ** 2)) * np.sqrt(np.sum((y - mean_y) ** 2))

    # 避免分母为零
    if denominator == 0:
        return 0
    return numerator / denominator

def column_pearson_corr(matrix1, matrix2):
    """
    计算两个矩阵的列间 Pearson 相关系数

    参数:
    matrix1, matrix2 -- 相同形状的二维 numpy 数组 (m×n)

    返回:
    相关系数矩阵 -- 形状为 (1, n) 的 numpy 数组
    """
    # 确保矩阵形状相同
    assert matrix1.shape == matrix2.shape, "矩阵形状必须相同"

    # 中心化矩阵
    center1 = matrix1 - np.mean(matrix1, axis=0, keepdims=True)
    center2 = matrix2 - np.mean(matrix2, axis=0, keepdims=True)

    # 计算分子 (协方差求和)
    numerator = np.sum(center1 * center2, axis=0)

    # 计算分母 (标准差乘积)
    denominator = np.sqrt(np.sum(center1 ** 2, axis=0)) * np.sqrt(np.sum(center2 ** 2, axis=0))

    # 处理分母为零的情况 (设为0避免NaN)
    denominator[denominator == 0] = np.inf

    # 计算相关系数
    corr = numerator / denominator

    # 返回行向量 (1×n)
    return corr.reshape(1, -1)

def distance(plaintext,key):
    product = key * 1729%3329
    if plaintext + product > 3329:
        temp = plaintext + product - 3329
    else:
        temp = plaintext + product
    hwc= bin(temp).count('1')
    if plaintext>product:
        temp2 = plaintext-product
    else :
        temp2 = 3329 - product + plaintext
    hwd= bin(temp2).count('1')

    hwproduct = bin(product).count('1')
    hwa = bin(plaintext).count('1')
    
    # return hwc
    # return  hwc + hwd
    # return  0.45*hwc + 0.55*hwd 
    return 2*hwa + hwc + hwd + hwproduct




def process_key_wrapper(args):
    """包装函数，用于处理单个密钥"""
    key, power_trace_mat, plaintext_list = args
    return process_key(key, power_trace_mat, plaintext_list)


def process_key(key, power_trace_mat, plaintext_list):
    """处理单个密钥的函数（独立于类）"""
    sample_number = power_trace_mat.shape[1]
    plaintext_mat = np.zeros((len(plaintext_list), sample_number))

    for index, plaintext in enumerate(plaintext_list):
        h = distance(plaintext, key)
        plaintext_mat[index, :] = h

    return key, column_pearson_corr(power_trace_mat, plaintext_mat)


class CPA:
    def __init__(self, power_trace_file,base_file=None, sample_number=5000, plaintext_number=3329, key_number=3329,
                 process_number=None,
                 low_sample = None,
                 high_sample = None):
        self.power_trace_file = power_trace_file
        self.sample_number = sample_number
        self.key_number = key_number
        self.plaintext_number = plaintext_number
        self.process_number = process_number or max(1, mp.cpu_count() - 1)

        if low_sample is not None:
            self.low_sample = low_sample
        else:
            self.low_sample = 0
        
        if high_sample is not None:
            self.high_sample = high_sample
        else :
            self.high_sample = sample_number
        
        self.sample_number = self.high_sample - self.low_sample

        self.plaintext_list = []
        self.power_trace_mat = None
        self.base_power = None
        if base_file is not None:
            with open(base_file,'r') as bf:
                base_power_str = bf.readline()
                self.base_power = np.array(base_power_str.strip().split(', ')).astype(np.float64)
        

    def read_power(self):
        """读取功耗轨迹数据"""
        self.power_trace_mat = np.zeros((self.plaintext_number, self.sample_number))

        with tq(total=self.plaintext_number, desc="📊 Reading Power traces") as read_bar:
            with open(self.power_trace_file, 'r') as pf:
                number = 0
                for line in pf:
                    if number >= self.plaintext_number or not line.strip():
                        break
                    try:
                        plaintext_str, power_trace_str = line.split(':', 1)
                        plaintext = int(plaintext_str)
                        power_trace = np.array(power_trace_str.strip().split()).astype(np.float64)
                        #power_trace = np.array([p if p > 0 else -p for p in power_trace])
                        power_trace = power_trace[self.low_sample:self.high_sample]
                        #self.power_trace_mat[number, :] = power_trace
                        self.power_trace_mat[plaintext, :] = power_trace
                        self.plaintext_list.append(plaintext)
                        number += 1
                        read_bar.update(1)
                    except Exception as e:
                        print(f"Error parsing line: {line.strip()} - {str(e)}")

        # 确保数组大小正确
        if number < self.plaintext_number:
            self.power_trace_mat = self.power_trace_mat[:number, :]
            self.plaintext_number = number

        print(f"Successfully read {len(self.plaintext_list)} power traces")

    def analyze(self,output_file=None):
        """并行分析所有密钥"""
        print(f"🚀 Starting parallel CPA analysis with {self.process_number} processes...")

        # 准备任务参数
        tasks = [(key, self.power_trace_mat, self.plaintext_list)
                 for key in range(self.key_number)]

        self.result = {}

        # 使用进程池并行处理
        with Pool(processes=self.process_number) as pool:
            # 使用imap_unordered获取结果（无序但更快）
            with tq(total=self.key_number, desc="🔑 Analyzing keys") as pbar:
                for key, corr in pool.imap_unordered(process_key_wrapper, tasks, chunksize=10):
                    self.result[key] = corr
                    pbar.update(1)

                    # 每处理100个密钥更新一次进度
                    if pbar.n % 100 == 0:
                        pbar.set_postfix(processed=f"{pbar.n}/{self.key_number}")
        if output_file:
            with open(output_file,'w') as of:
                json.dump(self.result, of, ensure_ascii=False, indent=4,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else TypeError) 
        print('✅ CPA analysis completed successfully!')
        return self.result

    def draw_result(self, highlight_keys=None, zoom_range=None, save_path=None, result_file=None,show_max=False):
        """
        可视化 CPA 分析结果

        参数:
        highlight_keys: 需要突出显示的密钥列表
        zoom_range: 要放大的样本范围 (start, end)
        save_path: 图像保存路径
        """
        if (not hasattr(self, 'result') or not self.result) and not result_file:
            print("⚠️ 请先运行 analyze() 方法获取结果")
            return

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
        if result_file:
            with open(result_file,'r') as f:
                result = json.load(f)
            all_corrs = np.array([np.array(result[str(key)]).flatten() for key in range(self.key_number)])
        else :
            all_corrs = np.array([self.result[key].flatten() for key in range(self.key_number)])
            #all_corrs = np.array([self.result[key].flatten() for key in range(1,self.key_number)])
        print('Data read finish')
        # 创建图形和坐标轴
        fig = plt.figure(figsize=(14, 8))
        
        index_max = np.argmax(np.abs(all_corrs))
        max_key = index_max//self.sample_number
        max_index = index_max - (index_max//self.sample_number)*self.sample_number
        print(f'max r {np.max(np.abs(all_corrs))},arg {index_max},-> key:{max_key}, index:{max_index}')
        key_max = index_max//self.sample_number
        if zoom_range:
            # 如果有缩放范围，创建两个子图：全局视图和放大视图
            ax1 = plt.subplot(2, 1, 1)  # 全局视图
            ax2 = plt.subplot(2, 1, 2)  # 放大视图
            axes = (ax1, ax2)
        else:
            # 否则只创建单个视图
            ax = plt.subplot(1, 1, 1)
            axes = (ax,)

        # 绘制所有密钥的相关系数曲线 (高性能方式)
        for ax in axes:
            # 使用透明浅色绘制所有曲线
            x = np.arange(self.sample_number)
            segments = np.array([np.column_stack([x, y]) for y in all_corrs])
            norm = plt.Normalize(0, len(all_corrs))
            lc = LineCollection(segments, cmap='Greys', norm=norm, alpha=0.1, linewidth=0.3)
            ax.add_collection(lc)

            # 设置坐标轴范围
            ax.set_xlim(0, self.sample_number)
            ax.set_ylim(-1, 1)  # 相关系数范围
            #ax.set_ylim(-0.5, 0.35)  # 相关系数范围

            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.6)

            # 添加标签
            ax.set_xlabel('samples index')
            ax.set_ylabel('correlation')
        # 创建高对比度颜色列表（避免蓝色）
        
        # 突出显示特定密钥
        if highlight_keys:
            print(f"highlight key: {highlight_keys}")
            #colors = plt.cm.tab10(np.linspace(0, 1, len(highlight_keys)))
            
            for ax in axes:
                for i, key in enumerate(highlight_keys):
                    if result_file:
                        corr = np.array(result[str(key)]).flatten()
                    else:
                        corr = self.result[key].flatten()
                    label = f'key {key}'
                    ax.plot(corr, color=high_contrast_colors[i%10], linewidth=2, alpha=0.9, label=label)
                if result_file:
                    corr_max = np.array(result[str(key_max)]).flatten()
                else:
                    corr_max = self.result[key_max].flatten()
                if show_max:
                    label_max = f'key max {key_max}' 
                    ax.plot(corr_max, color=high_contrast_colors[9], linewidth=2, alpha=0.9, label=label_max)
                # 添加图例
                ax.legend(loc='upper right')

        # 设置缩放视图范围
        if zoom_range:
            ax2.set_title(f'zoom in ({zoom_range[0]}-{zoom_range[1]} samples)')
            ax2.set_xlim(zoom_range)

            # 在全局视图中标记缩放区域
            ax1.axvspan(zoom_range[0], zoom_range[1], color='yellow', alpha=0.2)
            ax1.text(zoom_range[0], 0.9, 'zoom in region', fontsize=10,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        # 添加标题
        title = f'CPA result ({self.key_number} keys, {self.sample_number} samples)'
        if highlight_keys:
            title += f'\nhighlight key(s): {", ".join(map(str, highlight_keys))}'
        plt.suptitle(title, fontsize=14)

        plt.tight_layout()

        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 结果已保存至: {save_path}")
        else:
            plt.show()

    def draw_trace(self,trace_number=0):
        x = np.arange(self.sample_number)
        #plt.plot(x,self.power_trace_mat[trace_number,:]-self.base_power)
        
        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        # 设置坐标轴范围
        ax1.set_xlim(0, self.sample_number)
        ax1.set_ylim(-27000, 27000)  # 范围
        ax2.set_xlim(0, self.sample_number)
        ax2.set_ylim(-27000, 27000)  # 范围

        # 添加网格
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax2.grid(True, linestyle='--', alpha=0.6)

        # 添加标签
        ax1.set_xlabel('samples index')
        ax1.set_ylabel('correlation')
        ax2.set_xlabel('samples index')
        ax2.set_ylabel('correlation')
        label_base = 'base_power_trace'
        ax1.plot(self.base_power, color='#000FFD', linewidth=2, alpha=0.9, label=label_base)
        label_trace = f'power_trace{trace_number}'
        ax2.plot(self.power_trace_mat[trace_number,:], color='#00F00F',  linewidth=2, alpha=0.9, label=label_trace)
        
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        #plt.plot(x,self.base_power)
        plt.show()

    def analyze_one_process(self,output_file=None):
        self.result = {}
        plaintext_mat = np.zeros((self.power_trace_mat.shape))
        with tq(total=self.key_number, desc="🔑 Analyzing keys") as pbar:
            for key in range(self.key_number):
                for index, plaintext in enumerate(self.plaintext_list):
                    h = distance(plaintext, key)
                    plaintext_mat[index, :] = h
                self.result[key] = column_pearson_corr(plaintext_mat,self.power_trace_mat)
                pbar.update(1)
                # 每处理100个密钥更新一次进度
                if pbar.n % 100 == 0:
                    pbar.set_postfix(processed=f"{pbar.n}/{self.key_number}")
        if output_file:
            with open(output_file,'w') as of:
                json.dump(self.result, of, ensure_ascii=False, indent=4,
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else TypeError) 
        print('✅ CPA analysis completed successfully!')
        return self.result

    def analyze_one_key(self,output_file=None):
        plaintext_mat = np.zeros((self.power_trace_mat.shape))
        for index,plaintext in enumerate(self.plaintext_list):
            #h = bin(plaintext).count('1')
            if plaintext > 0:
                h = bin(plaintext^(plaintext-1)).count('1')
            else:
                h= bin(plaintext).count('1')
            plaintext_mat[index,:] = h
            if index % 300 == 0:
                print(f'Processed {index} traces')
        correlation = column_pearson_corr(plaintext_mat,self.power_trace_mat)
        t = np.arange(self.sample_number)
        plt.plot(t,correlation[0])
        plt.show()
    
    

        


if __name__ == "__main__":
    #power_file = "data/666/delta/aver_down20_delta.txt"
    #power_file = "data/average/delta_traces_loop_5_max.txt"
    #power_file = 'data/2773/average/average_loop_25_freq.txt'
    # power_file = 'data/2773/average/average_loop_25_align.txt'
    #power_file = 'data/2773/average/average_loop_25_log.txt'
    #power_file = 'data/BeforeData/ntt_pipeline_traces_cd.txt'
    # power_file = 'data/2773/average/average_loop_25.txt'
    #power_file = 'data/2773/average/cd_loop0_mean10.txt'
    # power_file = 'data/2773/average/cd_loop0_mean100.txt'
    #power_file = 'data/2773/delta/mean10_loop0_sub_base.txt'
    #power_file = 'data/mod_1ntt/666/average/average_cd_loop_2.txt'
    power_file = 'data/LNA7m/666/average/averaged-20-lnax7.txt'
    # power_file = 'data/666/source_cd_file/ntt_pipeline_traces-loop2.txt'
    #power_file = 'data/666/average/average_cd_loop_20.txt'
    #power_file = 'data/1234/average/average_cd_loop_32.txt'
    #power_file = 'data/2619/average/average_cd_loop_25.txt'
    #power_file = 'data/2773/source_cd_file/ntt_pipeline_traces-loop0.txt'
    #result_file = 'result/r1.txt'
    result_file = 'result/r2.txt'
    base_file = 'data/BeforeData/base_average.txt'
    s_num=1
    low_sample = 4300
    high_sample = 5000
    mode =2 # 1 analyze ;2 analyze and show result ;3 show power trace ; 4 show one key;5 show corelation by trace number ;else show result file
    cpa = CPA(
        power_trace_file=power_file,
        #base_file = base_file,
        sample_number=5000//s_num,
        plaintext_number=3329,
        key_number=3329,
        process_number=32,
        low_sample=low_sample,
        high_sample=high_sample
    )
    if mode == 1:
        cpa.read_power()
        result = cpa.analyze(
            output_file=result_file,
            )
    elif mode == 2:
        cpa.read_power()
        result = cpa.analyze()
        #result = cpa.analyze_one_process()
        cpa.draw_result(
            # highlight_keys=[2773],
            highlight_keys=[666],
            # highlight_keys=[1234],
            # highlight_keys=[2095],
            # highlight_keys=[2619],
            #highlight_keys=[2663],
            show_max = True
            #zoom_range=(100, 4000),
            #save_path='picture/cpa_result_01.png'
        )
    elif mode == 3 :
        cpa.read_power()
        cpa.draw_trace(112)
    elif mode == 4:
        cpa.read_power()
        cpa.analyze_one_key()
    else :
        cpa.draw_result(
            highlight_keys=[2773],
            #zoom_range=(100, 4000),
            save_path='picture/cpa_result_b_01.png',
            result_file=result_file,
            show_max = True
        )