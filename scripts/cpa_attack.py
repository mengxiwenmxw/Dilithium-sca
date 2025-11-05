import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pywt 
from scipy.signal import correlate 

########################
#      Parameters      #
########################


# TRACES_FILENAME = './result_preprocess/666-lnax5/averaged-20-lnax5.txt'
TRACES_FILENAME = '/15T/Projects/Dilithium-SCA/data/traces/2773/averaged/old_cym_scripts/averaged-0to19.txt'

special_b = 2773
FIG3_ENABLE = False  # 是否绘制fig3

wn = 1729
N = 8380417
B_GUESS_MIN = 0      # b_guess 的最小值 (包含)
B_GUESS_MAX = 3328  # b_guess 的最大值 (包含)

a_last = 0

########################
#  Preprocessing Setup #
########################
USE_DENOISING = False  # 是否启用小波降噪
DOWNSAMPLE_FACTOR = 20 # 降采样（建议在对齐和降噪后，如果需要的话，在main函数中进行）
ROI_START   = 3000 // DOWNSAMPLE_FACTOR
ROI_END     = 5000 // DOWNSAMPLE_FACTOR

CONFIG_WAVELET = 'db4'  # 小波类型
CONFIG_LEVEL = 8        # 小波分解层数
CONFIG_MODE = 'soft'

######################
#     PARSE DATA     #
######################
def parse_data(filename):
# def parse_data(filename, downsample_factor, downsample_method):
    """
    解析 {TRACES_FILENAME} 对应的文件。
    由于功耗数据可能跨越多行，此函数会正确处理。
    
    Args:
        filename: 数据文件名
        downsample_factor: 降采样因子, 每downsample_factor个点取一个值
        downsample_method: 降采样方法, 'mean'表示取平均值, 'max'表示取最大值
    """
    power_traces = {}
    parse_cnt = 0
    with open(filename, 'r') as f:
        current_trace_num = -1
        trace_data_list = []
        for line in f:
            line = line.strip()
            if not line:
                raise ValueError("ERROR: blank")
            plaintext, trace = line.split(':', 1)
            plaintext = int(plaintext)
            trace = np.array(trace.strip().split()).astype(np.float64)
            
            power_traces[plaintext] = trace
    return power_traces

#############################
#   PREPROCESSING FUNCTIONS   #
#############################

def plot_denoising_effect(original_trace, denoised_trace, result_dir, wavelet=CONFIG_WAVELET, level=CONFIG_LEVEL, mode=CONFIG_MODE):
    """
    可视化展示原始迹线和降噪后的迹线对比。
    """
    plt.figure(figsize=(15, 6))
    plt.plot(original_trace, label='Original Trace', color='blue', alpha=0.7)
    plt.plot(denoised_trace, label='Denoised Trace', color='red', alpha=0.7)
    plt.title('Denoising Effect Comparison')
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f'{wavelet}_{level}_{mode}_effect.png'), dpi=300)

def denoise_traces(traces_matrix, wavelet=CONFIG_WAVELET, level=CONFIG_LEVEL, mode=CONFIG_MODE):
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
            coeffs = pywt.wavedec(trace, wavelet, level=level)
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
            new_coeffs.append(pywt.threshold(c, threshold, mode=mode))
            
        # 4. 重构信号
        denoised_traces[i] = pywt.waverec(new_coeffs, wavelet)

    print("--- 小波降噪完成 ---")
    return denoised_traces

def popcount(n):
    """计算一个整数的汉明重量 (二进制表示中'1'的个数)"""
    # 确保处理负数时与硬件的位宽行为一致
    if n < 0:
        return bin(n & 0xFFFFFF).count('1') # 假设最大位宽为24位
    return bin(n).count('1')

# --- 从Verilog文件提取的ROM数据 --- #
ROM_H_TABLE = [0, 3270, 3211, 3152, 3093, 3034, 2975, 2916, 2857, 2798, 2739, 2680, 2621, 2562, 2503, 2444]
ROM_M_TABLE = [0, 2285, 1241, 197, 2482, 1438, 394, 2679, 1635, 591, 2876, 1832, 788, 3073, 2029, 985]
ROM_L_TABLE = [0, 767, 1534, 2301, 3068, 506, 1273, 2040, 2807, 245, 1012, 1779, 2546, 3313, 751, 1518]


def calculate_product(b):
    product = (b * wn) % N
    return product

def HD(num1,num2):
    return bin(num2^num1).count('1')

#######################
#     CALCULATION     #
#######################
def calculate_correlation_for_guess(b_guess, traces_matrix, a_values):
    """
    Used to calculate single b_guess's Pearson correlation coe
    """
    num_traces = len(a_values)
    product_guess = calculate_product(b_guess)
    global a_last
    ##### get model L #####

    ### 1 ###
    L = np.zeros(num_traces)
    file_path = "/15T/Projects/Dilithium-SCA/data/special_files/Random_3000.txt"
    for i in range(num_traces):

    # ## MODEL 1 ##
    #     c_bit_width  = 13
    #     a_val = a_values[i]
    #     c_sum = a_val + product_guess
    #     c_val = c_sum - N if c_sum >= N else c_sum
    #     # L[i] = c_bit_width - popcount(c_val)
    #     L[i] = popcount(c_val)

    ## MODEL Dilithium
        a_idx = a_values[i]
        with open(file_path, 'r') as f:
            all_lines = f.readlines()
        start_index = a_idx
        end_index = a_idx + 6
        target_lines = all_lines[start_index:end_index]
        numbers = [int(line.strip()) for line in target_lines]

        a1, a2, a3, a4, a5, a6 = numbers
        L[i] = HD(a1* b_guess %N, a2* b_guess %N) + HD(a1* b_guess %N, a_last* b_guess %N)
        a_last = a6
    
    # ## MODEL 2 ##
    #     a_val = a_values[i]
    #     d_val = a_val - product_guess if a_val > product_guess else a_val - product_guess + N
    #     L[i] = popcount(d_val)

    # ## MODEL 3 ##
    #     a_val = a_values[i]
    #     c_sum = a_val + product_guess
    #     c_val = c_sum - N if c_sum >= N else c_sum
    #     d_val = a_val - product_guess if a_val > product_guess else a_val - product_guess + N
    #     L[i] = popcount(c_val) + popcount(d_val)

    # ## MODEL 4 ##
    #     # L向量不再依赖于a，对于一个b_guess，它是一个常数向量
    #     # 模拟的是product值本身写入寄存器时的功耗
    #     a_val = a_values[i]
    #     intermediate_sum = a_val + product_guess
    #     L[i] = popcount(intermediate_sum)

    if L.std() == 0: 
        return None, None

    mean_L = L.mean()
    std_L = L.std()
        
    # calculate dimension 0 (vertical)
    mean_traces = traces_matrix.mean(axis=0)
    std_traces = traces_matrix.std(axis=0)
        
    # discard points that do not change in 3329
    valid_indices = np.where(std_traces != 0)[0] # np.where returns a tuple, pick the first element
    if len(valid_indices) == 0:
        return None, None
        
    cov = np.mean((traces_matrix[:, valid_indices] - mean_traces[valid_indices]) * (L[:, np.newaxis] - mean_L), axis=0)
    corrs = cov / (std_L * std_traces[valid_indices])

    return corrs, valid_indices

##############
#    PLOT    #
##############
def plot_fig1 (traces_matrix, keys_to_plot_np, a_values, result_dir): 
    """
    绘制所有猜测密钥的相关系数随时间变化的图，并高亮显示特定密钥。
    """
    # ----- figure 1: Correlation vs Time ----- #
    print(f"INFO: Generating figure 1...")

    plt.figure(figsize=(15, 8))

    highlight_list = list(keys_to_plot_np)
    # Choose whether add special_b
    if B_GUESS_MIN <= special_b <= B_GUESS_MAX and special_b not in highlight_list:
        highlight_list.append(special_b)

    # --- Draw Background --- #
    print("INFO: Plotting background correlation curves...")
    for b_guess in range(B_GUESS_MIN, B_GUESS_MAX + 1):
        if b_guess in highlight_list:
            continue    # skip

        if(b_guess - B_GUESS_MIN) % 200 == 0:
            print(f">>> Plotting background curve: {b_guess}/{B_GUESS_MAX}")

        corrs, valid_indices = calculate_correlation_for_guess(b_guess, traces_matrix, a_values)
        if corrs is not None and len(valid_indices) > 0:
            # Use grey slim transparent line to draw background
            plt.plot(valid_indices, corrs, color='lightgray', linewidth=0.5, alpha=0.7, zorder=1)


    print(f"INFO: Plotting highlighted correlation curves...")
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(highlight_list))) # 为5条曲线选择不同颜色
    for i, b_guess in enumerate(highlight_list):
        corrs, valid_indices = calculate_correlation_for_guess(b_guess, traces_matrix, a_values)
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

            if ROI_START is not None and ROI_END is not None: 
                mask = (valid_indices >= ROI_START) & (valid_indices <= ROI_END)
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
    
    fig1_path = os.path.join(result_dir, 'fig1_corrs_over_time.png')
    plt.savefig(fig1_path, dpi=300)
    print(f"[+] 图1已保存至: {fig1_path}")
    plt.close()

def plot_fig2 (max_corrs, keys_to_plot_np, result_dir):
    """
    绘制每个猜测密钥的最大相关系数图。
    """
        # ----- figure 2: Correlation vs b_guess ----- #
    print(f"INFO: Generating figure 2...")

    plt.figure(figsize=(15, 8))
    # 定义绘图的x轴范围和y轴数据
    b_range_to_plot = np.arange(B_GUESS_MIN, B_GUESS_MAX + 1)
    corrs_to_plot = max_corrs[B_GUESS_MIN : B_GUESS_MAX + 1]
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
    fig2_path = os.path.join(result_dir, 'fig2_cpa_result.png')
    plt.savefig(fig2_path, dpi=300)
    print(f"[+] 图2已保存至: {fig2_path}")
    plt.close()
    
def plot_fig3(traces_matrix, a_values, correct_key, result_dir):
    """
    绘制相关性随迹线数量变化的CPA收敛图 (fig3)。
    修正：确保追踪的峰值时间点 t_peak 在 ROI 内选取。
    """
    print("\nINFO: Generating figure 3 (Correlation vs. Number of Traces)...")
    print("WARNING: This process is computationally intensive and may take a long time.")

    N_STEP = 10

    num_traces = traces_matrix.shape[0]

    # --- 修正逻辑 START ---
    # 1. 使用全部迹线，在ROI内为正确密钥找到相关性最高的“峰值时间点”
    print("INFO: Step 1/3 - Finding the peak correlation time point within ROI...")
    full_corrs, full_valid_indices = calculate_correlation_for_guess(correct_key, traces_matrix, a_values)
    if full_corrs is None:
        print("ERROR: Cannot calculate correlation for the correct key. Aborting fig3 generation.")
        return
    
    t_peak = -1
    # 优先在 ROI 内寻找峰值
    if ROI_START is not None and ROI_END is not None:
        print(f"INFO: Using ROI [{ROI_START}, {ROI_END}] to find the peak time.")
        mask = (full_valid_indices >= ROI_START) & (full_valid_indices <= ROI_END)
        indices_in_roi = full_valid_indices[mask]
        corrs_in_roi = full_corrs[mask]

        if corrs_in_roi.size > 0:
            # 在 ROI 内的相关性数据中找到最大值的索引
            peak_idx_in_roi = np.argmax(corrs_in_roi)
            # 从 ROI 的索引中映射回真实的时间点
            t_peak = indices_in_roi[peak_idx_in_roi]
        else:
            print("WARNING: ROI is defined, but no valid data points were found within it for the correct key.")
    
    # 如果没有定义 ROI 或 ROI 内没有数据，则回退到寻找全局峰值
    if t_peak == -1:
        print("WARNING: Falling back to finding the global peak time (no ROI or ROI was empty).")
        peak_idx_global = np.argmax(full_corrs)
        t_peak = full_valid_indices[peak_idx_global]
    # --- 修正逻辑 END ---
    
    print(f"INFO: Peak time point selected for tracking: t = {t_peak}")

    # 2. 迭代计算 (此部分逻辑保持不变)
    print("INFO: Step 2/3 - Iteratively calculating correlations...")
    correlations_over_n = np.zeros((B_GUESS_MAX + 1, num_traces))
    start_n = 10 
    total_traces = traces_matrix.shape[0]
    for n in range(start_n, num_traces, N_STEP):
        if n % 100 == 0:
            print(f"  Processing with a random set of {n}/{total_traces} traces...")
        
        random_indices = np.random.choice(total_traces, n, replace=False)
        traces_subset = traces_matrix[random_indices, :]
        a_values_subset = a_values[random_indices]

        for b_guess in range(B_GUESS_MIN, B_GUESS_MAX + 1):
            corrs, valid_indices = calculate_correlation_for_guess(b_guess, traces_subset, a_values_subset)
            
            if corrs is not None:
                idx_map = np.where(valid_indices == t_peak)[0]
                if idx_map.size > 0:
                    correlations_over_n[b_guess, n] = corrs[idx_map[0]]
    
    # 3. 绘制图像 (此部分逻辑保持不变)
    print("INFO: Step 3/3 - Plotting the results...")
    plt.figure(figsize=(15, 8))
    n_values_to_plot = np.arange(start_n, num_traces, N_STEP)

    for b_guess in range(B_GUESS_MIN, B_GUESS_MAX + 1):
        if b_guess == correct_key:
            continue
        plt.plot(n_values_to_plot, correlations_over_n[b_guess, n_values_to_plot], color='gray', alpha=0.2, linewidth=0.8)

    plt.plot(n_values_to_plot, correlations_over_n[correct_key, n_values_to_plot], color='red', linewidth=1)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=1.5, label=f'Correct Key (b={correct_key})'),
        Line2D([0], [0], color='gray', lw=0.8, label='Incorrect Keys')
    ]

    plt.title('Correlation vs. Number of Traces')
    plt.xlabel('Number of Traces')
    plt.ylabel('Correlation')
    plt.legend(handles=legend_elements)
    plt.grid(True)
    plt.xlim(left=0)

    fig3_path = os.path.join(result_dir, 'fig3_cpa_convergence.png')
    plt.savefig(fig3_path, dpi=300)
    print(f"[+] 图3已保存至: {fig3_path}")
    plt.close()

#################
#    RUN CPA    #
#################
def run_cpa(traces_matrix, result_dir):
# def run_cpa(traces_dict, result_dir):
    """
    运行相关性功耗分析的完整流程。
    """
    # ----- CHECK INPUT & CONVERT TO NUMPY ----- #
    # 假设 a 的值是从 0 到 N-1，共N条轨迹
    # num_traces = N
    # if not traces_dict or 0 not in traces_dict or len(traces_dict[0]) == 0:
    #     print("功耗数据为空或格式错误。")
    #     return None, None
    
    # # 5000 points
    # num_samples = len(traces_dict[0])
    # print(f"num_samples: {num_samples}")

    # # 将字典转换为numpy数组以便进行高效的数学运算
    # # N * 5000
    # traces_matrix = np.zeros((num_traces, num_samples), dtype=np.float32)
    # a: 0 - 3328
    a_values = np.arange(num_traces)
    # for i in range(num_traces):
    #     if i in traces_dict:
    #         traces_matrix[i, :] = traces_dict[i]
            
    # dump max for each guess b
    max_corrs = np.zeros(B_GUESS_MAX + 1) # big enough
    
    print("INFO: >>> Start CPA... >>>")

    # ----- TRAVERSE ALL POSSIBLE B ----- #
    # 遍历'b'的所有N_guess个可能密钥猜测
    for b_guess in range(B_GUESS_MIN, B_GUESS_MAX + 1):
        if b_guess > B_GUESS_MIN and b_guess % 256 == 0:
            print(f">>> 正在测试 b_guess = {b_guess}/{B_GUESS_MAX}...")
        corrs, valid_indices = calculate_correlation_for_guess(b_guess, traces_matrix, a_values)
        if corrs is not None:
            # 找出最大的相关系数值（不取绝对值）
            # max_corrs[b_guess] = np.max(np.abs(corrs))
            if ROI_START is not None and ROI_END is not None: 
                mask = (valid_indices >= ROI_START) & (valid_indices <= ROI_END)

                corrs_in_roi = corrs[mask]

                if corrs_in_roi.size > 0 :
                    max_corrs[b_guess] = np.max(corrs_in_roi)
                else:
                    print(f"ERROR: corrs_in_roi is null")
                    max_corrs[b_guess] = 0
        else:
            max_corrs[b_guess] = np.max(corrs)

    # --- Analyse Results --- #
    # 仅在已测试的范围内查找相关性最高的猜测值
    # NumPy 数组的 Slicing 功能, 起始索引(包含):结束索引(不包含), relevant_corrs = numpy.ndarray
    relevant_corrs = max_corrs[B_GUESS_MIN : B_GUESS_MAX + 1]
    # argsort返回的是在切片内的索引，因此需要加上范围的起始值
    # 升序排列得到一个索引列表, 取最后五个并反转, :: 表示选取所有元素, 而 -1 的 step 表示从后向前遍历
    top_5_indices_relative = np.argsort(relevant_corrs)[-5:][::-1]
    top_5_indices = top_5_indices_relative + B_GUESS_MIN

    # top_5_indices = np.argsort(max_corrs)[-5:][::-1] # 从高到低排序
    correct_b_guess = top_5_indices[0]
    print("\n[+] --- CPA Complete ---")
    print(f"[+] INFO: Most likely 'b' is: {correct_b_guess}, max corrs: {max_corrs[correct_b_guess]:.4f}")
    print(f"INFO: Top 5 likely and their corrs:")
    for i in top_5_indices:
        print(f"\tb_guess = {i:<4} | Max Correlation = {max_corrs[i]:.4f}") # why not 1-3 here?

    # ---------- PLOT ---------- #
    plot_fig1 (traces_matrix, top_5_indices, a_values, result_dir)
    plot_fig2 (max_corrs, top_5_indices, result_dir)
    if FIG3_ENABLE:
        try:
            from cpa_fig3_smooth import plot_fig3_multiprocess_smooth
            plot_fig3_multiprocess_smooth(traces_matrix, a_values, correct_b_guess, result_dir, num_processes=6, special_b=special_b)
        except ImportError as e:
            print(f"ERROR: Failed to import multiprocess version")
    # else:
    #     plot_fig3(traces_matrix, a_values, correct_b_guess, result_dir)

# # --- 根据配置选择使用单进程还是多进程版本绘制fig3_random --- #
    # if USE_MULTIPROCESS_FIG3:
    #     try:
    #         from cpa_fig3_multiprocess import plot_fig3_multiprocess
    #         plot_fig3_multiprocess(traces_matrix, a_values, correct_b_guess, result_dir, 
    #                              show_interactive=True, save_data=True)
    #     except ImportError as e:
    #         print(f"WARNING: Failed to import multiprocess version: {e}")
    #         print("INFO: Falling back to single-process version.")
    #         plot_fig3(traces_matrix, a_values, correct_b_guess, result_dir)
    # else:
    #     plot_fig3(traces_matrix, a_values, correct_b_guess, result_dir)

##############
#    MAIN    #
##############
if __name__ == "__main__":
    try:
        # 确保已安装所需库: pip install numpy matplotlib

        # --- Make Dir --- #
        timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
        tag_with_txt = TRACES_FILENAME.split("/")[3]
        tag = tag_with_txt.split(".")[0]
        result_dir = os.path.join("/15T/Projects/Dilithium-SCA/result", timestamp+"-"+tag)
        os.makedirs(result_dir, exist_ok=True)
        # print(f"本次运行结果将保存在: {result_dir}")
        # --- Load and Run CPA --- #
        # 使用降采样功能，每10个点取平均值或最大值
        # downsample_method可以是'mean'或'max'或'min'

        # --- 加载原始数据 ---
        all_traces_dict = parse_data(TRACES_FILENAME)
        if not all_traces_dict:
            raise ValueError("未能从数据文件中加载任何功耗迹线。")
        print(f"成功加载 {len(all_traces_dict)} 条原始功耗迹线。")
            # 5000 points
        num_samples = len(all_traces_dict[0])
        print(f"num_samples: {num_samples}")

        # 将字典转换为numpy矩阵以便处理
        num_traces = N
        num_samples = len(all_traces_dict[0])
        traces_matrix = np.zeros((num_traces, num_samples), dtype=np.float32)
        for i in range(num_traces):
            if i in all_traces_dict:
                traces_matrix[i, :] = all_traces_dict[i]
        
        # --- 执行可选的预处理步骤 ---
        processed_traces = traces_matrix
        # if USE_ALIGNMENT:
        #     processed_traces = align_traces(processed_traces)
        
        if USE_DENOISING:
            processed_traces = denoise_traces(processed_traces, wavelet=CONFIG_WAVELET, level=CONFIG_LEVEL, mode=CONFIG_MODE)
            plot_denoising_effect(traces_matrix[special_b-1], processed_traces[special_b-1], result_dir)
            print(f"Denoising >>> processed_traces.shape: {processed_traces.shape}")

        # (可选) 在所有处理后进行降采样
        if DOWNSAMPLE_FACTOR > 1:
            print(f"\n--- 正在执行最后一步: 降采样 (因子: {DOWNSAMPLE_FACTOR}) ---")
            num_complete = processed_traces.shape[1] // DOWNSAMPLE_FACTOR
            processed_traces = processed_traces[:, :num_complete * DOWNSAMPLE_FACTOR].reshape(num_traces, num_complete, DOWNSAMPLE_FACTOR).max(axis=2)
            print("--- 降采样完成 ---")

        # --- 运行CPA ---
        run_cpa(processed_traces, result_dir) # 注意: run_cpa 现在需要接收一个矩阵


    except FileNotFoundError:
        print("错误: 未找到 {TRACES_FILENAME}。请确保脚本与数据文件在同一目录。")
    except Exception as e:
        print(f"发生了一个未知错误: {e}")