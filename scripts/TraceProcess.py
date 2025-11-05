from re import T
from typing import Tuple
import numpy as np
from numpy.lib.function_base import average
from tqdm import tqdm as tq
import matplotlib.pyplot as plt
import os 
from scipy.signal import correlate
import pywt
import math

class MkDir:
    def __init__(self,data_root,key_number,power_file_number,file_name='mau_traces_loop',tag:str='') -> None:
        self.data_root = data_root
        self.key_number = key_number
        self.key_root_path = os.path.join(self.data_root , str(self.key_number)+tag +'/')
        self.tag = tag
        self.power_file_number = power_file_number
        self.file_name = file_name

    def mk_dir(self):
        print(f">>> Start to mk dir for key {self.key_number} in {self.key_root_path}:\n\
            -{str(self.key_number)+ self.tag}\n\
                |--power_traces\n\
                |--averaged\n\
                    |--none\n\
                    |--align\n\
                    |--denoise\n\
                    |--align-denoise\n\
            ")
        power_traces_path = os.path.join(self.key_root_path , 'power_traces/')
        os.makedirs(power_traces_path,exist_ok=True)
        for sub_dir in ['none/','align/','denoise/','align-denoise/']:
            averaged_path = os.path.join(self.key_root_path , 'averaged/' , sub_dir )
            os.makedirs(averaged_path,exist_ok=True)
        print(f">>> Create Dirs finish , pls cp power traces files to  {self.key_root_path}/power_traces/")
    

    def get_power_traces_files(self):
        power_files = []
        for i in range(self.power_file_number):
            now_file = os.path.join(self.key_root_path,'power_traces/',self.file_name + str(i) + '.txt')
            if os.path.isfile(now_file):
                power_files.append(now_file)
            else:
                raise ValueError(f"path:{now_file} is not a file")

        return power_files

def visualize_preprocessing(unaligned, aligned, averaged, a_val, result_dir):
    """生成并保存用于展示预处理效果的对比图。"""
    print(f"--- 正在为 a = {a_val} 生成可视化图表 ---")
    
    # --- 图1: 对齐效果对比 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=True)
    
    # 只画前5条曲线以便观察
    num_curves_to_show = min(5, unaligned.shape[0])

    ax1.set_title(f'Before Alignment (First {num_curves_to_show} Traces)', fontsize=14)
    for i in range(num_curves_to_show):
        ax1.plot(unaligned[i, :], label=f'Loop {i}')
    ax1.legend()
    
    ax2.set_title(f'After Alignment (First {num_curves_to_show} Traces)', fontsize=14)
    for i in range(num_curves_to_show):
        ax2.plot(aligned[i, :], label=f'Loop {i}')
    
    fig1.suptitle(f'Comparison of Alignment (a = {a_val})', fontsize=16, weight='bold')
    plt.xlabel("Time", fontsize=12)
    fig1.supylabel("Power", fontsize=12)
    
    fig1_path = os.path.join(result_dir, f'fig_alignment_a{a_val}.png')
    plt.savefig(fig1_path)
    print(f"[+] 对齐效果图已保存至: {fig1_path}")
    plt.close(fig1)

    # --- 图2: 平均效果 ---
    fig2, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制所有对齐后的曲线
    for i in range(aligned.shape[0]):
        ax.plot(aligned[i, :], color='grey', alpha=0.3)
    
    # 绘制平均后的曲线
    ax.plot(averaged, color='red', linewidth=2, label='Averaged Trace')
    
    ax.set_title(f'Average Effect (a = {a_val})', fontsize=16, weight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.legend()

    fig2_path = os.path.join(result_dir, f'fig_averaging_a{a_val}.png')
    plt.savefig(fig2_path)
    print(f"[+] 平均效果图已保存至: {fig2_path}")
    plt.close(fig2)

    # --- 图3: 原始曲线概览 ---
    print(f"--- 正在为 a = {a_val} 生成图3: 单独绘制每条对齐后的曲线 ---")
    
    num_plots = aligned.shape[0]
    # 定义网格布局，这里设置为每行最多5个子图
    plots_per_row = 5
    num_rows = math.ceil(num_plots / plots_per_row)

    fig3, axes = plt.subplots(num_rows, plots_per_row, figsize=(15, 3 * num_rows), squeeze=False)
    
    # 将axes数组扁平化以便于索引
    axes_flat = axes.flatten()

    # 循环绘制每一条对齐后的曲线到独立的子图中
    for i in range(num_plots):
        ax = axes_flat[i]
        ax.plot(aligned[i, :], color='teal')
        ax.set_ylim(-20000, 25000)
        ax.set_title(f'Aligned Trace from Loop {i}', fontsize=10)
        # 为每个子图设置坐标轴标签，可以简化一些
        ax.set_xlabel("Time", fontsize=8)
        ax.set_ylabel("Power", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # 隐藏多余的、未使用的子图轴
    for i in range(num_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig3.suptitle(f'All Individual Aligned Traces (a = {a_val})', fontsize=16, weight='bold')
    # 使用 tight_layout 自动调整子图参数，使其填充整个图像区域
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以防止主标题重叠

    fig3_path = os.path.join(result_dir, f'fig_all_aligned_a{a_val}.png')
    plt.savefig(fig3_path, dpi=300)
    print(f"[+] 所有对齐曲线的独立图已保存至: {fig3_path}")
    plt.close(fig3)

GENERATE_VISUALIZATIONS = True
A_VAL_TO_VISUALIZE = 2773

class TraceProcess:
    def __init__(self, sample_number=5000, plaintext_number=3329,
                save_root = None,
                save_file_name = None,
                align_feature_window : Tuple[int,int]=(None,None),
                align_max_shift: int = None,
                wavelet: str = 'db4',
                wt_level: int = 8,
                wt_mode : str = 'soft',
                down_mode: str = 'mean',
                down_num: int = 20
                ):
        if save_root is None or save_file_name is None:
            raise ValueError("ERROR: Need save root path and file name.")
       

        self.power_trace_files = None
        self.save_root= save_root
        self.save_name = save_file_name
        self.power_file_num = 0
        self.sample_number = sample_number
        self.plaintext_number = plaintext_number
        self.power_trace_mats = {}

        if align_feature_window[0] is None or align_feature_window[1] is None or align_max_shift is None:
            raise ValueError("ERROR: pls set align window and align max shift.")
        else :
            self.feature_window_start,self.feature_window_end = align_feature_window[0],align_feature_window[1]
            print(f"feature_window = {self.feature_window_start}, {self.feature_window_end}")
            self.max_shift = align_max_shift
        self.wavelet = wavelet
        self.level = wt_level
        self.mode  = wt_mode
        if down_mode in ['max','min','mean']:
            self.down_mode = down_mode
            self.down_num = down_num

        else:
            raise ValueError("ERROR: parameter down mode must one of 'max','mean','min'")
        
    def read_power(self,power_trace_files=None,):
        """读取功耗轨迹数据"""
        if power_trace_files is None:
            raise ValueError("ERROR: Need power traces files")
        self.power_trace_files = power_trace_files
        self.power_file_num = len(power_trace_files)
        
        for file_num in range(self.power_file_num):
            power_trace_mat = np.zeros((self.plaintext_number, self.sample_number))
            with tq(total=self.plaintext_number, desc=f">>> Reading Power traces:{self.power_trace_files[file_num]}") as read_bar:
                with open(self.power_trace_files[file_num], 'r') as pf:
                    number = 0
                    for line in pf:
                        if number >= self.plaintext_number or not line.strip():
                            break
                        try:
                            plaintext_str, power_trace_str = line.split(':', 1)
                            plaintext = int(plaintext_str)
                            power_trace = np.array(power_trace_str.strip().split()).astype(np.int64)

                            if len(power_trace) < self.sample_number:
                                power_trace = np.pad(power_trace, (0, self.sample_number - len(power_trace)))
                            elif len(power_trace) > self.sample_number:
                                power_trace = power_trace[:self.sample_number]

                            #power_trace_mat[number, :] = power_trace
                            power_trace_mat[plaintext, :] = power_trace
                            number += 1
                            read_bar.update(1)
                        except Exception as e:
                            print(f"Error parsing line: {line.strip()} - {str(e)}")

            # 确保数组大小正确
            if number < self.plaintext_number:
                power_trace_mat = power_trace_mat[:number, :]
                self.plaintext_number = number

            print(f"Successfully read {file_num+1} power traces")
            self.power_trace_mats[file_num]=power_trace_mat
        print(f">>> Successfuly read {self.power_file_num} power files")

    def align_traces(self,traces_matrix):
        """
        使用基于特征窗口的互相关方法对齐一组迹线。
        该方法只对 feature_window_start 到 feature_window_end 之间的片段进行互相关，
        以一个已知的、独特的峰值作为对齐的“锚点”。
        
        :param traces_matrix: 输入的功耗迹线矩阵
        :param feature_window_start: 特征窗口的起始索引
        :param feature_window_end: 特征窗口的结束索引
        :param max_shift: 在特征窗口内允许的最大位移，用于增加稳定性
        :return: 对齐后的功耗迹线矩阵
        """
        num_traces, num_samples = traces_matrix.shape
        if num_traces <= 1:
            return traces_matrix
            
        aligned_traces = np.zeros_like(traces_matrix)
        
        # 1. 创建参考特征：不再是参考整个曲线，而是参考所有曲线在特征窗口内的平均波形
        reference_feature = np.mean(traces_matrix[:, self.feature_window_start:self.feature_window_end], axis=0)
        
        freqs = np.fft.fftfreq(num_samples)

        for i in range(num_traces):
            if i % 500 == 0:
                print(f"正在对齐迹线: {i}/{num_traces-1}...")
                
            current_trace = traces_matrix[i]
            
            # 2. 从当前曲线中提取相同的特征窗口
            current_feature = current_trace[self.feature_window_start:self.feature_window_end]
            
            # 3. 只对特征窗口进行互相关
            cross_corr = correlate(current_feature, reference_feature, mode='full')
            
            # 4. 在互相关结果中寻找最佳位移（仍然建议在小范围内搜索以策万全）
            center_index = len(reference_feature) - 1 # 注意：这里的center_index是基于短的feature的长度
            
            search_start = max(0, center_index - self.max_shift)
            search_end = min(len(cross_corr), center_index + self.max_shift + 1)
            
            windowed_corr = cross_corr[search_start:search_end]
            shift_in_window = np.argmax(windowed_corr)
            
            shift = (search_start + shift_in_window) - center_index
            
            # 5. 将计算出的位移 shift 应用于完整的功耗曲线
            current_fft = np.fft.fft(current_trace)
            phase_shift = np.exp(-2j * np.pi * freqs * shift)
            corrected_fft = current_fft * phase_shift
            
            aligned_traces[i] = np.fft.ifft(corrected_fft).real
            
        return aligned_traces

    def denoise_traces(self,traces_matrix):
        """
        使用小波变换对每条迹线进行降噪。
        原理参考论文 4.3.3 节。
        """
        print("\n--- 正在执行预处理第二步: 小波降噪 ---")
        num_traces = traces_matrix.shape[0]
        denoised_traces = np.zeros_like(traces_matrix)
        
        for i in range(num_traces):
            if i % 500 == 0:
                print(f">>> 正在降噪迹线: {i}/{num_traces-1}...")
            
            trace = traces_matrix[i]
            # 1. 小波分解
            try:
                coeffs = pywt.wavedec(trace, self.wavelet, level=self.level)
            except Exception as e:
                print(f"小波分解失败: {e}")
                continue
            
            # 2. 计算阈值 (VisuShrink)
            # 噪声标准差估计
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(trace)))
            
            # 3. 对细节系数应用软阈值
            new_coeffs = [coeffs[0]] # 保留近似系数
            for c in coeffs[1:]:
                new_coeffs.append(pywt.threshold(c, threshold, mode=self.mode))
                
            # 4. 重构信号
            denoised_traces[i] = pywt.waverec(new_coeffs, self.wavelet)

        print("--- 小波降噪完成 ---")
        return denoised_traces


    def down_sample_one_plainttext(self,traces_matrix):
        if self.down_mode == 'mean':
            return traces_matrix.reshape(-1,self.down_num).mean(axis=1)
        elif self.down_mode == 'max':
            return traces_matrix.reshape(-1,self.down_num).max(axis=1)
        else:
            return traces_matrix.reshape(-1,self.down_num).min(axis=1)
        

    def process_traces(self,power_trace_files=None,mode = None,down=False,down_num=None):
        """
        mode : string = none, align, denoise, align-denoise
        down : bool 
        down_num : int
        Trace process and Save result. 
        """
     
        pre_fix = 'none' if mode is None else mode
        suffix = f'{len(power_trace_files)}_down{down_num}.txt' if down else f'{len(power_trace_files)}.txt'
        save_path = os.path.join(self.save_root ,pre_fix + '/')
        save_file_name = self.save_name + suffix

        ### read all:
        self.read_power(power_trace_files=power_trace_files)
        averaged_traces = {}
        align_flag = True if mode == 'align' or mode == 'align-denoise' else False
        
        denoise_flag = True if mode == 'denoise' or mode == 'align-denoise' else False
        
        down_flag = down 
        for plaintext in range(self.plaintext_number):
            if plaintext > 0 and plaintext % 250 == 0:
                print(f"正在处理 a = {plaintext}/{self.plaintext_number-1}...")
            
            traces_for_current_a = np.zeros((self.power_file_num, self.sample_number))
            for file_idx in range(self.power_file_num):
                traces_for_current_a[file_idx, :] = self.power_trace_mats[file_idx][plaintext,:]
                
            # align
            aligned_batch = self.align_traces(traces_for_current_a) if align_flag else traces_for_current_a 
            # denoise
            denoise_batch = self.denoise_traces(aligned_batch) if denoise_flag else aligned_batch
            # average
            average_trace_plaintext = np.mean(denoise_batch, axis=0)
            # down
            down_trace    = self.down_sample_one_plainttext(average_trace_plaintext) if down_flag else average_trace_plaintext
            averaged_traces[plaintext] = down_trace
            # --- 如果是指定的a值，则生成可视化图表 ---
            if GENERATE_VISUALIZATIONS and plaintext == A_VAL_TO_VISUALIZE:
                visualize_preprocessing(traces_for_current_a, aligned_batch, down_trace, plaintext, save_path)
        print(f">>> Save processed file to {save_path}")
        
        with tq(total=self.plaintext_number, desc=f"Writing Power file:{save_file_name}") as read_bar:
            with open(os.path.join(save_path, save_file_name),'w') as wf:
                for plaintext , averaged_trace in averaged_traces.items():
                    wf.write(str(plaintext)+':'+str(averaged_trace.tolist()).replace(' ','').replace(',',' ')[1:-1]+'\n')
                    read_bar.update(1)
        print('Processing power traces finish.')
    
    def save_average_lna_power_trace(self,output_file=None,max_value=50000,low_sample=4300,high_sample=5000):
        
        if output_file is None:
            raise ValueError('Need a output file')
        power_traces_dict = {}
        power_trace_number_dict = {}
        file_first = True
        for power_file in self.power_trace_files:
            with tq(total=self.plaintext_number, desc=f"Reading Power file:{power_file}") as read_bar:
                with open(power_file,'r') as pf:
                    for line in pf:
                        plaintext_str , value_str = line.split(':',1)
                        power_trace = np.array(value_str.strip().split()).astype(np.int64)
                        if file_first:
                            if np.max(np.abs(power_trace[low_sample:high_sample])) < max_value:
                                power_traces_dict[plaintext_str] = power_trace
                                power_trace_number_dict[plaintext_str] = 1
                            else:
                                power_traces_dict[plaintext_str] = np.array([0 for _ in range(self.sample_number)])
                                power_trace_number_dict[plaintext_str] = 0
                        else :
                            if np.max(np.abs(power_trace[low_sample:high_sample])) < max_value:
                                power_traces_dict[plaintext_str] += power_trace
                                power_trace_number_dict[plaintext_str] += 1
                        read_bar.update(1)
            if(file_first):
                file_first = False
        
        with tq(total=self.plaintext_number, desc=f"Writing Power file:{output_file}") as read_bar:
            with open(output_file,'w') as wf:
                for plaintext_str , sum_power_trace in power_traces_dict.items():
                    average_trace = sum_power_trace/power_trace_number_dict[plaintext_str] if power_trace_number_dict[plaintext_str] else sum_power_trace
                    wf.write(plaintext_str+':'+str(average_trace.tolist()).replace(',',' ')[1:-1]+'\n')
                    read_bar.update(1)
        print('Processing power traces finish.')
    

    def down_sample(self,input_file,output_file,s_num=1,mode = 'mean'):
        if mode not in ['max','mean','min']:
            raise ValueError('var "mode" is wrong')
        power_trace_dict = {}
        with tq(total=self.plaintext_number, desc=f"Reading file:{input_file}") as read_bar:
            with open(input_file,'r') as cdf:
                for line in cdf:
                    plaintext_str , value_str = line.split(':',1)
                    power_trace = np.array(value_str.strip().split()).astype(np.float64)
                    if mode == 'max':
                        power_trace_dict[plaintext_str] = power_trace.reshape(-1,s_num).max(axis=1)
                    elif mode == 'mean':
                        power_trace_dict[plaintext_str] = power_trace.reshape(-1,s_num).mean(axis=1)
                    elif mode == 'min':
                        power_trace_dict[plaintext_str] = power_trace.reshape(-1,s_num).min(axis=1)
                    read_bar.update(1)
        with tq(total=self.plaintext_number, desc=f"Writing delta Power file:{output_file}") as read_bar:
            with open(output_file,'w') as wdpf:
                for plaintext_str , power_trace in power_trace_dict.items():
                    wdpf.write(plaintext_str+':'+str(power_trace.tolist()).replace(',',' ')[1:-1]+'\n')
                    read_bar.update(1)

    def show_trace(self,trace_file,trace_number=0,s_num=1):
        x = np.arange(self.sample_number//s_num)
        trace = None
        with open(trace_file,'r') as pf:
            for line in pf:
                plaintext_str, power_trace_str = line.split(':', 1)
                plaintext = int(plaintext_str)
                if plaintext == trace_number:
                    trace = np.array(power_trace_str.strip().split()).astype(np.float64)
                    break
        if trace is None:
            raise ValueError(f'trace {trace_number} is not in file:{trace_file}')
        plt.plot(x,trace)
        plt.show()

if __name__ == "__main__":
    pass