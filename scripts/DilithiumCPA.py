from CpaAttack import CPA,Draw
from datetime import datetime
import os


"""
 This file is used to get cpa result form averaged data / source data.
"""
special_b = 2773
DIR_TAG = '_kyber'
DATA_ROOT = f"/15T/Projects/Dilithium-SCA/data/traces/{special_b}/averaged/"

### file path
# DATA_ROOT = f"/15T/Projects/Dilithium-SCA/data/traces/{special_b}/averaged/"
# TAG = ""
# TRACE_FILE_NAME = "averaged_mau_loop20.txt"
# TAG = "align/"
TAG = "old_cym_scripts-3329/"
TRACE_FILE_NAME = "averaged-0to19.txt"
# TRACE_FILE_NAME = "averaged_mau_loop20.txt"

#RANDOM_FILE = "/15T/Projects/Dilithium-SCA/data/special_files/Random_3000.txt"
RANDOM_FILE = "/15T/Projects/Dilithium-SCA/data/special_files/random_3000_0-3328.txt"

PICTURE_PATH = "/15T/Projects/Dilithium-SCA/result/"
trace_file = DATA_ROOT+ TAG + TRACE_FILE_NAME
if not os.path.isfile(trace_file):
    raise ValueError(f"ERROR: cant find file {trace_file}")
if not os.path.isfile(RANDOM_FILE):
    raise ValueError(f"ERROR: cant find file {RANDOM_FILE}")


### vars
SAMPLE_NUM = 5000
PLAINTEXT_NUM = 2994
# PLAINTEXT_NUM = 3329
KEY_NUM  = 3329
PROCESS_NUM = 32
LOW_SAMPLE = 0
HIGH_SAMPLE = 5000
SAMPLE_NUM_RESULT = HIGH_SAMPLE - LOW_SAMPLE



### instance
cpa = CPA(
    power_trace_file=trace_file,
    random_plaintext_file=RANDOM_FILE,
    sample_number=SAMPLE_NUM,
    traces_number=PLAINTEXT_NUM,
    key_number=KEY_NUM,
    process_number=PROCESS_NUM,
    low_sample= LOW_SAMPLE,
    high_sample=HIGH_SAMPLE
)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H:%M")
    bak = ''
    time_tag = timestamp+'-'+TAG[:-1]+bak
    cpa.read_power()
    result = cpa.analyze()
    draw = Draw(
        picture_save_path=PICTURE_PATH,
        key_number=KEY_NUM,
        sample_number=cpa.sample_number
    )
    #print(result)
    top_5_keys = draw.get_top_key(result=result,abs=False)
    print(f"Top 5 keys:\n{top_5_keys}")

    # draw.draw_result(
    #     result=result,
    #     highlight_keys=[special_b]
    # )
    
    draw.draw_fig1(
        result=result,
        keys_to_plot_np=top_5_keys,
        special_b=special_b,
        time_tag=time_tag
    )
    draw.draw_fig2(
        result=result,
        keys_to_plot_np=top_5_keys,
        special_b=special_b,
        time_tag=time_tag
    )
    