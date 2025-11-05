import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math

########################
#      Parameters      #
########################


# SOURCE_DIR = 'traces_666'
# 功耗曲线文件的命名格式 (用{}表示变化的数字)
FILE_PATTERN = 'mau_traces-loop{}.txt'

# --- 文件范围设置 ---
# 起始文件编号 (例如：loop5.txt 对应 START_FILE_NUM = 5)
START_FILE_NUM = 0
# 结束文件编号 (例如：loop10.txt 对应 END_FILE_NUM = 10，包含此文件)
END_FILE_NUM = 19
# 计算要处理的文件总数
NUM_FILES = END_FILE_NUM - START_FILE_NUM + 1

# 每个文件中的迹线条数
NUM_TRACES_PER_FILE = 2994
# 处理后输出的新文件名
OUTPUT_FILENAME = f'averaged-{START_FILE_NUM}to{END_FILE_NUM}.txt'

# --- 可视化设置 ---
# 是否生成可视化对比图
GENERATE_VISUALIZATIONS = True
# 要进行可视化对比的'a'值 (选择一个有代表性的即可)
A_VAL_TO_VISUALIZE = 2773

# 包含源功耗曲线文件的目录
SOURCE_DIR = f"/15T/Projects/Dilithium-SCA/data/traces/{A_VAL_TO_VISUALIZE}_kyber/power_traces/"


############################
#     HELPER FUNCTIONS     #
############################

def parse_data_single_file(filename):
    """解析单个功耗数据文件。"""
    power_traces = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                plaintext, trace = line.split(':', 1)
                plaintext = int(plaintext)
                trace = np.array(trace.strip().split()).astype(np.float64)
                power_traces[plaintext] = trace
            except ValueError as e:
                print(f"警告: 解析文件 '{filename}' 时出错: '{line}'. 错误: {e}. 已跳过此行。")
    return power_traces

def align_traces(traces_matrix, feature_window_start=330, feature_window_end=340, max_shift=7):
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
    reference_feature = np.mean(traces_matrix[:, feature_window_start:feature_window_end], axis=0)
    
    freqs = np.fft.fftfreq(num_samples)

    for i in range(num_traces):
        if i % 500 == 0:
            print(f"正在对齐迹线: {i}/{num_traces-1}...")
            
        current_trace = traces_matrix[i]
        
        # 2. 从当前曲线中提取相同的特征窗口
        current_feature = current_trace[feature_window_start:feature_window_end]
        
        # 3. 只对特征窗口进行互相关
        cross_corr = correlate(current_feature, reference_feature, mode='full')
        
        # 4. 在互相关结果中寻找最佳位移（仍然建议在小范围内搜索以策万全）
        center_index = len(reference_feature) - 1 # 注意：这里的center_index是基于短的feature的长度
        
        search_start = max(0, center_index - max_shift)
        search_end = min(len(cross_corr), center_index + max_shift + 1)
        
        windowed_corr = cross_corr[search_start:search_end]
        shift_in_window = np.argmax(windowed_corr)
        
        shift = (search_start + shift_in_window) - center_index
        
        # 5. 将计算出的位移 shift 应用于完整的功耗曲线
        current_fft = np.fft.fft(current_trace)
        phase_shift = np.exp(-2j * np.pi * freqs * shift)
        corrected_fft = current_fft * phase_shift
        
        aligned_traces[i] = np.fft.ifft(corrected_fft).real
        
    return aligned_traces

# def align_traces(traces_matrix):
#     """使用基于互相关和FFT相位校正的方法对齐一组迹线。"""
#     num_traces, num_samples = traces_matrix.shape
#     if num_traces <= 1:
#         return traces_matrix
        
#     aligned_traces = np.zeros_like(traces_matrix)
    
#     reference_trace = traces_matrix[0]
#     aligned_traces[0] = reference_trace
    
#     reference_fft = np.fft.fft(reference_trace)
#     freqs = np.fft.fftfreq(num_samples)

#     for i in range(1, num_traces):
#         if i % 500 == 0:
#             print(f"正在对齐迹线: {i}/{num_traces-1}...")
#         current_trace = traces_matrix[i]
#         cross_corr = correlate(current_trace, reference_trace, mode='full')
#         shift = np.argmax(cross_corr) - (num_samples - 1)
        
#         current_fft = np.fft.fft(current_trace)
#         phase_shift = np.exp(-2j * np.pi * freqs * shift)
#         corrected_fft = current_fft * phase_shift
        
#         aligned_traces[i] = np.fft.ifft(corrected_fft).real
        
#     return aligned_traces

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


##############
#    MAIN    #
##############
if __name__ == "__main__":
    try:
        print("--- 开始功耗迹线预处理 ---")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H.%M")
        result_dir = f"/15T/Projects/Dilithium-SCA/data/traces/{A_VAL_TO_VISUALIZE}/averaged/old_cym_scripts-3329/"
        os.makedirs(result_dir, exist_ok=True)
        print(f"本次运行结果将保存在: {result_dir}")

        # --- 1. 加载所有数据文件 ---
        all_data = []
        print(f"正在从 '{SOURCE_DIR}' 目录加载文件 loop{START_FILE_NUM} 到 loop{END_FILE_NUM} (共 {NUM_FILES} 个文件)...")
        for i in range(START_FILE_NUM, END_FILE_NUM + 1):
            print(f">>> Processing loop{i} trace")
            filename = os.path.join(SOURCE_DIR, FILE_PATTERN.format(i))
            traces_dict = parse_data_single_file(filename)
            if not traces_dict: raise ValueError(f"文件 {filename} 为空或无法解析。")
            all_data.append(traces_dict)
        print("INFO: All files loaded")

        # --- 2. 逐个'a'值进行对齐和平均 ---
        print("\n--- 开始对齐与平均过程 ---")
        num_samples = len(all_data[0][0])
        averaged_traces = {}

        for a_val in range(NUM_TRACES_PER_FILE):
            if a_val > 0 and a_val % 250 == 0:
                print(f"正在处理 a = {a_val}/{NUM_TRACES_PER_FILE-1}...")
            
            traces_for_current_a = np.zeros((NUM_FILES, num_samples))
            for file_idx in range(NUM_FILES):
                traces_for_current_a[file_idx, :] = all_data[file_idx][a_val]
                # print(all_data[file_idx][a_val])
                # raise ValueError("Pause")
            
            aligned_batch = align_traces(traces_for_current_a)
            averaged_trace = np.mean(aligned_batch, axis=0)
            averaged_traces[a_val] = averaged_trace

            # --- 如果是指定的a值，则生成可视化图表 ---
            if GENERATE_VISUALIZATIONS and a_val == A_VAL_TO_VISUALIZE:
                visualize_preprocessing(traces_for_current_a, aligned_batch, averaged_trace, a_val, result_dir)
        
        print("对齐与平均处理完成。")
        
        # --- 3. 将结果写入新文件 ---
        output_path = os.path.join(result_dir, OUTPUT_FILENAME)
        print(f"\n--- 正在将结果写入到 '{output_path}' ---")
        with open(output_path, 'w') as f:
            for a_val in range(NUM_TRACES_PER_FILE):
                trace_str = ' '.join(map(str, averaged_traces[a_val]))
                f.write(f"{a_val}:{trace_str}\n")
        
        print(f"[+] 成功！高质量的平均功耗曲线已保存至 '{output_path}'。")

    except FileNotFoundError:
        print(f"错误: 目录 '{SOURCE_DIR}' 或其中的文件未找到。请确保脚本与数据目录在正确的位置。")
    except ImportError:
        print("错误: 缺少必要的库。请运行 'pip install numpy scipy matplotlib'")
    except Exception as e:
        print(f"发生了一个未知错误: {e}")