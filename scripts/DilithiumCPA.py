from CpaAttack import CPA,Draw
from datetime import datetime
import os
import json


"""
 This file is used to get cpa result form averaged data / source data.
"""
# import json file
try:
    with open("setting.json") as setf:
        config = json.load(setf)
except:
    raise ValueError("Cant find setting.json file")

general = config.get("General",{})
path = config.get("PATH",{})
config_cpa = config.get("CPA",{})
config_process = config.get("PROCESS",{})
# import end


special_b = general.get("KEY_TRUE")

DIR_TAG = general.get("DIR_TAG")
DATA_ROOT = path.get("DATA_ROOT")
TRACE_FILE_NAME = config_process.get("AVERAGED_FILE_PREFIX_NAME") + str(general.get("FILE_NUM")) + '.txt'


### file path

TRACE_PROCESS_MODE = general.get("TRACE_PROCESS_MODE")
remark = general.get("REMARK")


#RANDOM_FILE = "/15T/Projects/Dilithium-SCA/data/special_files/Random_3000.txt"
RANDOM_ROOT = path.get("SPECIAL_ROOT")
RANDOM_NAME = config_cpa.get("RANDOM_FILE_NAME")
RANDOM_FILE = os.path.join(RANDOM_ROOT,RANDOM_NAME)

PICTURE_PATH = path.get("RESULT_ROOT")

trace_file = os.path.join(DATA_ROOT,f"{special_b}{DIR_TAG}/averaged/{TRACE_PROCESS_MODE}/",TRACE_FILE_NAME)

if not os.path.isfile(trace_file):
    raise ValueError(f"ERROR: cant find file {trace_file}")
if not os.path.isfile(RANDOM_FILE):
    raise ValueError(f"ERROR: cant find file {RANDOM_FILE}")


### vars
SAMPLE_NUM = general.get("SAMPLE_NUM")
PLAINTEXT_NUM = general.get("PLAINTEXT_NUM")
# PLAINTEXT_NUM = 3329
KEY_NUM  = config_cpa.get("KEY_NUM")
PROCESS_NUM = config_cpa.get("PROCESS_NUM")


LOW_SAMPLE = config_cpa.get("LOW_SAMPLE")
HIGH_SAMPLE = config_cpa.get("HIGH_SAMPLE")
SAMPLE_NUM_RESULT = HIGH_SAMPLE - LOW_SAMPLE

DOWNSAMPLE_FACTOR = config_cpa.get("DOWNSAMPLE_FACTOR")



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
    
    time_tag = timestamp+'-'+str(special_b)+DIR_TAG+'-'+TRACE_PROCESS_MODE+remark
    
    cpa.read_power(down_sample_factor = DOWNSAMPLE_FACTOR )

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
    