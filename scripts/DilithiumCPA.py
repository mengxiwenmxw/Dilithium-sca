from CpaAttack import CPA
from CpaAttack import Draw
from TraceProcess import TraceProcess

### file path
data_root = "/15T/Projects/Dilithium-SCA/data/traces/averaged/"
tag = "none/"
trace_file_name = "mau_loop20.txt"
random_file = "/15T/Projects/Dilithium-SCA/data/special_files/Random_3000.txt"
picture_path = "/15T/Projects/Dilithium-SCA/result/"
trace_file = data_root + tag + trace_file_name

### vars
sample_num = 5000
trace_num = 2994
key_num  = 3329
process_num = 32
low_sam = 0
high_sam = 5000

special_b = 2773

### instance
cpa = CPA(
    power_trace_file=trace_file,
    random_plaintext_file=random_file,
    sample_number=sample_num,
    traces_number=trace_num,
    key_number=key_num,
    process_number=process_num,
    low_sample= low_sam,
    high_sample=high_sam
)
draw = Draw(
    picture_save_path=picture_path,
    key_number=key_num,
    sample_number=sample_num
)





if __name__ == "__main__":
    
    cpa.read_power()
    result = cpa.analyze()
    
    top_5_keys = draw.get_top_key(result=result)

    draw.draw_result(
        result=result,
        highlight_keys=[special_b]
    )
    
    draw.draw_fig1(
        result=result,
        keys_to_plot_np=top_5_keys,
        special_b=special_b,
    )
    draw.draw_fig2(
        result=result,
        keys_to_plot_np=top_5_keys,
        special_b=special_b,
    )
    