from re import T
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm as tq
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os 

class TraceProcess:
    def __init__(self,power_trace_file=None, sample_number=5000, plaintext_number=3329,base=False):
        self.power_trace_file = power_trace_file
        self.sample_number = sample_number
        self.plaintext_number = plaintext_number


        self.plaintext_list = []
        self.power_trace_mat = None
        self.base = base

    def read_power(self):
        """读取功耗轨迹数据 base"""
        self.power_trace_mat = np.zeros((self.plaintext_number, self.sample_number))
        if not self.base:
             raise ValueError( 'Not base mode')
        with tq(total=self.plaintext_number, desc="📊 Reading Power traces") as read_bar:
            with open(self.power_trace_file, 'r') as pf:
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

    def average_base(self,output_file=None):
        mean_power = np.mean(self.power_trace_mat,axis=0)
        x = np.arange(self.sample_number)
        if output_file is not None:
            with open(output_file,'w') as bf:
                bf.write('0'+':'+str(mean_power.tolist())[1:-1].replace(',',' '))
        plt.plot(x,mean_power)
        plt.show()

    def save_average_power_trace(self,output_file=None):
        if self.base:
            raise ValueError('now mode: base')
        if output_file is None:
            raise ValueError('Need a output file')
        power_traces_dict = {}
        power_trace_file_number = 0
        for power_file in self.power_trace_file:
            with tq(total=self.plaintext_number, desc=f"Reading Power file:{power_file}") as read_bar:
                with open(power_file,'r') as pf:
                    for line in pf:
                        plaintext_str , value_str = line.split(':',1)
                        power_trace = np.array(value_str.strip().split()).astype(np.int64)
                        if power_trace_file_number == 0:
                            power_traces_dict[plaintext_str] = power_trace
                        else :
                            power_traces_dict[plaintext_str] += power_trace
                        read_bar.update(1)
            power_trace_file_number +=1
        
        with tq(total=self.plaintext_number, desc=f"Writing Power file:{output_file}") as read_bar:
            with open(output_file,'w') as wf:
                for plaintext_str , sum_power_trace in power_traces_dict.items():
                    average_trace = sum_power_trace/power_trace_file_number
                    wf.write(plaintext_str+':'+str(average_trace.tolist()).replace(',',' ')[1:-1]+'\n')
                    read_bar.update(1)
        print('Processing power traces finish.')
    
    def save_average_lna_power_trace(self,output_file=None,max_value=50000,low_sample=4300,high_sample=5000):
        if self.base:
            raise ValueError('now mode: base')
        if output_file is None:
            raise ValueError('Need a output file')
        power_traces_dict = {}
        power_trace_number_dict = {}
        file_first = True
        for power_file in self.power_trace_file:
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
    
    def save_delta_power_trace(self,average_cd_file,average_without_cd_file,delta_file,sample_number=1):
        delta_power_trace_dict = {}
        with tq(total=self.plaintext_number, desc=f"Reading Power cd file:{average_cd_file}") as read_bar:
            with open(average_cd_file,'r') as cdf:
                for line in cdf:
                    plaintext_str , value_str = line.split(':',1)
                    power_trace = np.array(value_str.strip().split()).astype(np.float64)
                    delta_power_trace_dict[plaintext_str] = power_trace.reshape(-1,sample_number).max(axis=1)
                    read_bar.update(1)
        with tq(total=self.plaintext_number, desc=f"Reading Power without cd file:{average_without_cd_file}") as read_bar:
            with open(average_without_cd_file,'r') as wcdf:
                for line in wcdf:
                    plaintext_str , value_str = line.split(':',1)
                    power_trace = np.array(value_str.strip().split()).astype(np.float64)
                    delta_power_trace_dict[plaintext_str] -= power_trace.reshape(-1,sample_number).max(axis=1)
                    read_bar.update(1)
        with tq(total=self.plaintext_number, desc=f"Writing delta Power file:{delta_file}") as read_bar:
            with open(delta_file,'w') as wdpf:
                for plaintext_str , delta_power_trace in delta_power_trace_dict.items():
                    wdpf.write(plaintext_str+':'+str(delta_power_trace.tolist()).replace(',',' ')[1:-1]+'\n')
                    read_bar.update(1)
        print('Processing delta power traces finish.')

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

    def sub_base(self,power_file,base_file,output_file,s_sum=1):
        with open(base_file,'r') as bf:
            for line in bf:
                _,base_trace_str = line.split(':',1)
                base_trace = np.array(base_trace_str.strip().split()).astype(np.float64)
        power_trace_dict = {}
        with tq(total=self.plaintext_number, desc=f"Reading file:{power_file}") as read_bar:
            with open(power_file,'r') as cdf:
                for line in cdf:
                    plaintext_str , value_str = line.split(':',1)
                    power_trace = np.array(value_str.strip().split()).astype(np.float64)
                    power_trace_dict[plaintext_str] = power_trace - base_trace
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
    data_root = 'data/LNA7m/'
    #modes = [0] 
    """ 0 mkdir ; 1 save average cd ; 2 save average without cd ; 
        3 save delta file ; 4 save aver down  sample delta file ;
        5 save delta down sample; 6 base file process ; 7 save average LNA traces;
        other flexiable mode""" 
    modes = [0]
    ###### config #####
    b_name = 666
    loop_cd_num= 20
    loop_wt_cd_num= 16
    down_sample_num = 20

    sample_number = 5000
    plaintext_number = 3329

    output_average_cd_file_name = 'average_cd_loop_'+str(loop_cd_num)
    output_average_wt_cd_file_name = 'average_wt_cd_loop_'+str(loop_wt_cd_num)
    output_aver_delta_file_name = 'delta_aver_cd_loop_'+str(loop_cd_num)
    output_aver_down_delta_file_name = 'aver_'+'down'+str(down_sample_num) + '_delta'
    output_aver_delta_down_file_name = 'aver_delta_'+'down'+str(down_sample_num)

    input_cd_file_name = 'ntt_pipeline_traces-loop' 
    input_wt_cd_file_name = 'ntt_pipeline_traces-loop'

    source_cd_file_path = data_root+str(b_name)+'/'+'source_cd_file/'
    source_wt_cd_file_path = data_root+str(b_name)+'/'+'source_wt_cd_file/'
    average_file_path = data_root+str(b_name)+'/'+'average/'
    delta_file_path = data_root+str(b_name)+'/' + 'delta/'
    
    power_files_cd =[]
    power_files_wt_cd = []

    for cd_file_num in range(loop_cd_num):
        power_files_cd.append(source_cd_file_path+input_cd_file_name+str(cd_file_num)+'.txt')
    for wt_cd_file_num in range(loop_wt_cd_num):
        power_files_wt_cd.append(source_wt_cd_file_path+input_wt_cd_file_name+str(wt_cd_file_num)+'.txt')

    #### FULL files' name ####
    average_cd_file = average_file_path + output_average_cd_file_name + '.txt'
    average_wt_cd_file = average_file_path + output_average_wt_cd_file_name + '.txt'
    delta_file = delta_file_path + output_aver_delta_file_name + '.txt'
    aver_down_delta_file = delta_file_path + output_aver_down_delta_file_name + '.txt'
    aver_delta_down_file = delta_file_path + output_aver_delta_down_file_name + '.txt'

    for mode in modes:
        ##### file path set #####
        if mode == 0:
            print('Make Dir')
            os.makedirs(source_cd_file_path,exist_ok=True)
            os.makedirs(source_wt_cd_file_path,exist_ok=True)
            os.makedirs(average_file_path,exist_ok=True)
            os.makedirs(delta_file_path,exist_ok=True)
            print('Finish.')
        elif mode == 1:
            print('Save average power with cd')
            tp_cd=TraceProcess(
                power_trace_file=power_files_cd,
                sample_number=sample_number,
                plaintext_number=plaintext_number,
            )
            tp_cd.save_average_power_trace(output_file=average_cd_file)
            # tp_cd.show_trace(
            #     trace_file=average_cd_file,
            #     s_num=1)
        elif mode == 2:
            print('Save average power without cd')
            tp_wt_cd = TraceProcess(
                power_trace_file=power_files_wt_cd,
                sample_number=sample_number,
                plaintext_number=plaintext_number,
            )
            tp_wt_cd.save_average_power_trace(output_file=average_wt_cd_file)
            # tp_wt_cd.show_trace(
            #     trace_file=average_wt_cd_file,
            #     s_num=1)
        
        elif mode in [3,4,5]:
            tp_delta = TraceProcess(
                sample_number=sample_number,
                plaintext_number=plaintext_number,
            )
            if mode ==3:
                print('Save delta power file ')
                tp_delta.save_delta_power_trace(
                    average_cd_file=average_cd_file,
                    average_without_cd_file=average_wt_cd_file,
                    delta_file=delta_file,
                    sample_number=1)
            elif mode ==4:
                print('Save aver down sample delta power file ')
                tp_delta.save_delta_power_trace(
                    average_cd_file=average_cd_file,
                    average_without_cd_file=average_wt_cd_file,
                    delta_file=aver_down_delta_file,
                    sample_number=down_sample_num)
            elif mode == 5:
                print('Save  delta power data down sample file')
                tp_delta.down_sample(
                    input_file=delta_file,
                    output_file=aver_delta_down_file,
                    s_num=sample_number,
                    mode= 'max')
        elif mode == 6:
            power_base = 'data/base/ntt_pipeline_traces_base.txt'
            base_output= 'data/base/base_average.txt'
            tp_base = TraceProcess(
                power_trace_file=power_base,
                sample_number=5000,
                plaintext_number=3329,
                base= True
            )
            tp_base.read_power()
            tp_base.average_base(output_file=base_output)
        elif mode == 7:
            print('save average LNA power trace ')
            tp_lna = TraceProcess(
                power_trace_file=power_files_cd,
                sample_number=sample_number,
                plaintext_number=plaintext_number,
            )
            tp_lna.save_average_lna_power_trace(output_file=average_cd_file,max_value=50000,low_sample=3000)
        else :
            #power_file = 'data/2773/source_cd_file/ntt_pipeline_traces-loop0.txt'
            #base_file = 'data/base/mean10_base.txt'
            #power_file = 'data/2773/average/cd_loop0_mean10.txt'
            #output_file = 'data/2773/average/cd_loop0_mean10.txt'
            #output_file = 'data/2773/average/cd_loop0_mean100.txt'
            #output_file = 'data/2773/average/cd_loop0_mean20.txt'
            #output_file = 'data/base/mean10_base.txt'
            #output_file = 'data/2773/delta/mean10_loop0_sub_base.txt'
            #output_file = 'data/mod_1ntt/666/source_cd_file/ntt_pipeline_traces-loop0.txt'
            #output_file = 'data/mod_1ntt/666/average/average_cd_loop_2.txt'
            output_file = 'data/LNA/666/source_cd_file/ntt_pipeline_traces-loop1.txt'
            #output_file = 'data/LNA/666/average/average_cd_loop_20_without_error.txt'
            #output_file = 'data/LNA/666/average/average_cd_loop_20.txt'
            s_num = 1
            tp = TraceProcess(
                sample_number=5000,
                plaintext_number=3329,
            )
            # tp.down_sample(
            #     input_file=power_file,
            #     output_file= output_file,
            #     s_num=s_num,
            #     mode='mean'
            # )
            # tp.sub_base(
            #     power_file=power_file,
            #     base_file=base_file,
            #     output_file=output_file,
            #     s_sum=s_num
            #     )
            tp.show_trace(trace_file=output_file,s_num=s_num,trace_number=112)
            