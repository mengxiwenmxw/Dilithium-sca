import os
from typing import Tuple
import numpy as np
from multiprocessing import Pool,shared_memory
from numpy.core.multiarray import ravel_multi_index
from tqdm import tqdm as tq
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import json
import linecache

matplotlib.use('Agg')

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

#### get the pearson coorelation between two matrix

def column_pearson_corr(matrix1, matrix2):
    """
    计算两个矩阵的列间 Pearson 相关系数
    参数:
    matrix1, matrix2 -- 相同形状的二维 numpy 数组 (mxn)
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

### get distance in cycle 7

a_last = 0
qNUM = 3329

########################################
#     Get Prediction Vector Method     #
########################################
def distance(plaintexts,key):
    global a_last
    p0 = 0
    for i in range(3):
        p0 += HD((plaintexts[2-i]*key),(plaintexts[3-i]*key))

    input_7 = HD(plaintexts[5], 0)
    input_6 = HD(plaintexts[4], plaintexts[5])
    input_5 = HD(plaintexts[3], plaintexts[4])
    input_4 = HD(plaintexts[2], plaintexts[3])
    input_3 = HD(plaintexts[1], plaintexts[2])
    input_2 = HD(plaintexts[0], plaintexts[1])
    input_1 = HD(a_last, plaintexts[0])

    mm_result = HD(plaintexts[0]*key%qNUM, plaintexts[1]*key%qNUM)
    output = HD((a_last*key)%qNUM, (plaintexts[0]*key)%qNUM)

    a_last = plaintexts[4]
    # return output + mm_result
    return output + mm_result

    # ##### Verify #####
    # wn = 1729
    # N = 3329
    # a_val = plaintexts[0]
    # product_guess = (key * wn) % N
    # c_sum = a_val + product_guess
    # c_val = c_sum - N if c_sum >= N else c_sum
    # return HD(c_val, 0)


def HD(num1,num2):
    return bin(num2^num1).count('1')


#### get the correlation by multiprocesses

def process_key_wrapper(args):
    """包装函数，用于处理单个密钥"""
    key, power_trace_mat, plaintext_list = args
    return process_key(key, power_trace_mat, plaintext_list)


def process_key(key, power_trace_mat, plaintext_list):
    """处理单个密钥的函数（独立于类）"""
    sample_number = power_trace_mat.shape[1]
    H_distance_mat = np.zeros((len(plaintext_list), sample_number))

    for index, plaintexts in enumerate(plaintext_list):
        
        h = distance(plaintexts, key)
        # if (key == 2773) & ((index % 500 == 0) | (index % 500 == 1)):
        #     print(f"--- [DEBUG] h = {h}; plaintext = {plaintexts}; key = {key}\n\t\toutput = {(plaintexts[0]*key)%qNUM}, {(plaintexts[1]*key)%qNUM}, {(plaintexts[2]*key)%qNUM}, {(plaintexts[3]*key)%qNUM}, {(plaintexts[4]*key)%qNUM}, {(plaintexts[5]*key)%qNUM}")
        H_distance_mat[index, :] = h

    return key, column_pearson_corr(power_trace_mat, H_distance_mat)


def get_plaintexts(file_path,trace_number,plaintext_num=6):
    plaintexts = []
    for i in range(plaintext_num):
        line = linecache.getline(file_path, trace_number+i+1).rstrip('\n') #从1开始(HERE)
        if not line or line.isspace():
            raise ValueError(f"Plaintexts file line num not enough or cant find file {file_path}")
        plaintexts.append(int(line)) 
    return plaintexts  

     # 降采样参数
class CPA:
    def __init__(self, power_trace_file, random_plaintext_file,
                 sample_number=5000, traces_number=3329, key_number=3329,
                 process_number=None,
                 low_sample = None,
                 high_sample = None,
                 ):

        self.power_trace_file = power_trace_file
        self.random_plaintext_file = random_plaintext_file
        self.sample_number = sample_number
        self.key_number = key_number
        self.traces_number = traces_number
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
        
    def read_power(self,down_sample_factor=1):
        """读取功耗轨迹数据"""
        self.power_trace_mat = np.zeros((self.traces_number, self.sample_number))
        # 进度条
        with tq(total=self.traces_number, desc=">>>>> 01 Reading Power traces") as read_bar:
            with open(self.power_trace_file, 'r') as pf:
                number = 0
                # Read traces into matrix
                for line in pf:
                    if number >= self.traces_number or not line.strip():
                        break
                    else:
                        plaintext_str, power_trace_str = line.split(':', 1)
                        plaintext_number = int(plaintext_str)
                        power_trace = np.array(power_trace_str.strip().split()).astype(np.float64)
                        # >>> Slice >>>
                        power_trace = power_trace[self.low_sample:self.high_sample]
                        # >>> Save to matrix >>>
                        self.power_trace_mat[plaintext_number, :] = power_trace
                        ## changed
                        # if plaintext_number:
                        #     current_plaintexts = get_plaintexts(self.random_plaintext_file,plaintext_number-1)
                        # else :
                        #     current_plaintexts = get_plaintexts(self.random_plaintext_file,plaintext_number)
                        current_plaintexts = get_plaintexts(self.random_plaintext_file,plaintext_number)
                        ## change end
                        self.plaintext_list.append(current_plaintexts)
                        # if number % 500 == 0:
                        #     print(f"\n--- [DEBUG] 轨迹计数: {number} (文件中的 plaintext_number: {plaintext_number}) ---")
                        #     print(f"--- [DEBUG] Plaintexts: {current_plaintexts}\n")
                        number += 1
                        read_bar.update(1)
                    
                
        # 确保数组大小正确
        if number < self.traces_number:
            self.power_trace_mat = self.power_trace_mat[:number, :]
            self.traces_number = number
        print(f"INFO: 1. Successfully read {len(self.plaintext_list)} power traces")
        print(f"INFO: 2. Power traces matrix size: ({self.power_trace_mat.shape[0]} x {self.power_trace_mat.shape[1]})")
        ##### DownSample #####
        if down_sample_factor > 1:
            print(f"\tINFO: Down Sampling activated, processing...")
            sample_downsize = self.power_trace_mat.shape[1] // down_sample_factor
            self.power_trace_mat = self.power_trace_mat[:, :sample_downsize * down_sample_factor].reshape(self.power_trace_mat.shape[0], sample_downsize, down_sample_factor).max(axis=2)
            self.sample_number = sample_downsize
            print(f"\tINFO: Resize into ({self.power_trace_mat.shape[0]} x {sample_downsize})")

    def analyze(self,output_file=None):
        """并行分析所有密钥"""
        print(f">>>>> 02 Starting parallel CPA analysis with {self.process_number} processes...")
        # 准备任务参数
        tasks = [(key, self.power_trace_mat, self.plaintext_list)
                 for key in range(self.key_number)]
        self.result = {}
        # 使用进程池并行处理
        with Pool(processes=self.process_number) as pool:
            # 使用imap_unordered获取结果（无序但更快）
            with tq(total=self.key_number, desc="    Analyzing keys") as pbar:
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
        print('\tINFO: CPA analysis completed successfully!')
        return self.result


class Draw:
    def __init__(self,picture_save_path,sample_number=5000,key_number=3329,
                top_key_num = 5,
                compare_window:Tuple[int,int]=(None,None),
                ) -> None:
        self.save_path = picture_save_path
        self.sample_number = sample_number
        self.key_number = key_number
        self.top_key_num = top_key_num
        self.compare_window = compare_window
        
    def get_top_key(self,result,abs=False):
        left_cor  = 0
        right_cor =-1
        if self.compare_window[0]  and self.compare_window[1] :
            left_cor,right_cor = self.compare_window[0],self.compare_window[1]
        print(f">> Compare range (max correlation) = ({left_cor},{right_cor})")
        max_cor = {}
        for key in range(self.key_number):
            max_cor[key] = np.max(np.abs(result[key][0][left_cor:right_cor])) if abs else np.max(result[key][0][left_cor:right_cor])
        sorted_items = sorted(max_cor.items(), key=lambda x: x[1], reverse=True)
        # 获取前 n 个键
        top_keys = [item[0] for item in sorted_items[:self.top_key_num]]
        return np.array(top_keys)


    def draw_result(self, result,highlight_keys=None, zoom_range=None, save_path=None,show_max=False):
        """
        可视化 CPA 分析结果
        参数:
        highlight_keys: 需要突出显示的密钥列表
        zoom_range: 要放大的样本范围 (start, end)
        save_path: 图像保存路径
        """
        print("📊 准备可视化结果...")
        all_corrs = np.array([result[key].flatten() for key in range(self.key_number)])
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
                    
                    corr = result[key].flatten()
                    label = f'key {key}'
                    ax.plot(corr, color=high_contrast_colors[i%10], linewidth=2, alpha=0.9, label=label)
                corr_max = result[key_max].flatten()
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

    def draw_fig1(self,result,keys_to_plot_np,time_tag,special_b=2773,key_min=0,key_max=3328,roi_start=None,roi_en=None):
        """
        绘制所有猜测密钥的相关系数随时间变化的图，并高亮显示特定密钥。
        """
        # ----- figure 1: Correlation vs Time ----- #
        print(f"INFO: Generating figure 1...")
     

        plt.figure(figsize=(15, 8))

        highlight_list = list(keys_to_plot_np)
        # Choose whether add special_b
        if key_min <= special_b <= key_max and special_b not in highlight_list:
            highlight_list.append(special_b)

        # --- Draw Background --- #
        print("INFO: Plotting background correlation curves...")
        for b_guess in range(key_min, key_max + 1):
            if b_guess in highlight_list:
                continue    # skip

            if(b_guess - key_min) % 200 == 0:
                print(f">>> Plotting background curve: {b_guess}/{key_max}")

            corrs, valid_indices = result[b_guess][0],[i for i in range(self.sample_number)]
            if corrs is not None and len(valid_indices) > 0:
                # Use grey slim transparent line to draw background
                # print(f">>> {len(valid_indices)}")
                plt.plot(valid_indices, corrs, color='lightgray', linewidth=0.5, alpha=0.7, zorder=1)


        print(f"INFO: Plotting highlighted correlation curves...")

        colors = plt.cm.viridis(np.linspace(0, 1, len(highlight_list))) # 为5条曲线选择不同颜色
        for i, b_guess in enumerate(highlight_list):
            corrs, valid_indices = result[b_guess][0],[i for i in range(self.sample_number)]
            if corrs is not None:
                # 特殊处理 b_guess = special_b 的样式
                if b_guess == special_b:
                    style_kwargs = {'color': 'red', 'linestyle': '--', 'zorder': 100, 'linewidth': 1, 'label': f'special_b = {b_guess}'}
                else:
                    style_kwargs = {'color': colors[i], 'zorder': 50+i, 'linewidth': 0.8, 'label': f'b = {b_guess}'}

                plt.plot(valid_indices, corrs, **style_kwargs)
                # --- 标注峰值 --- #
                # peak_idx_in_corrs = np.argmax(np.abs(corrs))
                # 找出最大的相关系数值（不取绝对值）
                indices_for_peak = valid_indices
                corrs_for_peak = corrs

                if roi_start is not None and roi_en is not None: 
                    mask = (valid_indices >= roi_start) & (valid_indices <= roi_en)
                    indices_for_peak = valid_indices[mask]
                    corrs_for_peak = corrs[mask]

                if corrs_for_peak.size > 0:

                    peak_idx_in_corrs = np.argmax(corrs_for_peak)
                    x_peak = indices_for_peak[peak_idx_in_corrs]
                    y_peak = corrs_for_peak[peak_idx_in_corrs]
                    plt.annotate(f'({x_peak}, {y_peak:.3f})', 
                                 xy=(x_peak, y_peak), 
                                 xytext=(x_peak, y_peak + 0.03),
                                 ha='center',
                                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                                 zorder=101)

        plt.title('Pearson Coefficient vs. Time')
        plt.xlabel('Time')
        plt.ylabel('Correlation Coefficient (rho)')
        plt.legend()
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)

        # 设置纵轴范围 (可根据需要调整)
        plt.ylim(-0.2, 0.2)  # 例如：设置纵轴范围为 -0.5 到 0.5
        time_path = os.path.join(self.save_path,time_tag+'/')
        os.makedirs(time_path,exist_ok=True)
        fig1_path = os.path.join(self.save_path,time_tag+'/', 'fig1_corrs_over_time.png')
        plt.savefig(fig1_path, dpi=300)
        print(f"[+] 图1已保存至: {fig1_path}")
        plt.close()
    
    def draw_fig2(self,result,keys_to_plot_np,time_tag,special_b=2773,key_min=0,key_max=3328,roi_start=None,roi_en=None):
        """
        绘制每个猜测密钥的最大相关系数图。
        """
            # ----- figure 2: Correlation vs b_guess ----- #
        print(f"INFO: Generating figure 2...")

        plt.figure(figsize=(15, 8))
        # 定义绘图的x轴范围和y轴数据
        max_corrs = [np.max(result[key][0]) for key in range(self.key_number)]
        b_range_to_plot = np.arange(key_min, key_max + 1)
        corrs_to_plot = max_corrs[key_min : key_max + 1]
        plt.plot(b_range_to_plot, corrs_to_plot, alpha=0.6, label='All b_guess correlation')
        # plt.plot(range(N_guess), max_corrs, alpha=0.6, label='All b_guess correlation')

        # 标注Top 5的点
        for b_guess in keys_to_plot_np:
            y_val = max_corrs[b_guess]
            plt.plot(b_guess, y_val, 'bo', markersize=8, zorder=10) # 'ro' = red circle
            plt.annotate(f'({b_guess}, {y_val:.4f})',
                         xy=(b_guess, y_val),
                         xytext=(b_guess, y_val + 0.01),
                         ha='center',
                         fontsize=9,
                         zorder=11)

        # 特殊标注 b_guess = special_b
        if special_b not in keys_to_plot_np:
            y_val = max_corrs[special_b]
            plt.plot(special_b, y_val, 'ro', markersize=8, zorder=10) # 'bo' = blue circle
            plt.annotate(f'({special_b}, {y_val:.4f})',
                         xy=(special_b, y_val),
                         xytext=(special_b, y_val + 0.01),
                         ha='center',
                         color='blue',
                         fontsize=9,
                         zorder=11)


        plt.title('Every guess of \'b\'\'s maximum correlation coefficient')
        plt.xlabel('\'b\'\'s guess value')
        plt.ylabel('Maximum absolute correlation coefficient')
        plt.legend([plt.Line2D([0], [0], color='w')], [f'Found key: b={keys_to_plot_np[0]}']) # 简化图例
        plt.grid(True)
        time_path = os.path.join(self.save_path,time_tag+'/')
        os.makedirs(time_path,exist_ok=True)
        fig2_path = os.path.join(self.save_path,time_tag+'/', 'fig2_cpa_result.png')
        plt.savefig(fig2_path, dpi=300)
        print(f"[+] 图2已保存至: {fig2_path}")
        plt.close()

if __name__ == "__main__":
    pass