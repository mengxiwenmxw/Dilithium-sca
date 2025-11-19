from re import T
from TraceProcess import TraceProcess ,MkDir
import argparse
import json

"""
    This scripts is used to process power traces of Dilithium poly module.

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

KEY_TRUE = general.get("KEY_TRUE")
DIR_TAG = general.get("DIR_TAG")

FILE_NUM = general.get("FILE_NUM")



SAMPLE_NUM = general.get("SAMPLE_NUM")
PLAINTEXT_NUM = general.get("PLAINTEXT_NUM")

PROCESS_MODE = general.get("TRACE_PROCESS_MODE") # process mode: none (None) , align ('align'), denoise ('denoise'), align-denoise ('align-denoise')
DOWN = True if config_process.get("DOWN_IN_PROCESS") == "True" else False # bool 
DOWN_NUM = config_process.get("DOWN_FACTOR")


DATA_ROOT = path.get("DATA_ROOT")
SOURCE_FILE_PREFIX_NAME = config_process.get("SOURCE_FILE_PREFIX_NAME")

SAVE_ROOT = DATA_ROOT +f'{KEY_TRUE}{DIR_TAG}/averaged/'
SAVE_FILE_NAME = config_process.get("AVERAGED_FILE_PREFIX_NAME")

ALIGN_WINDOW = (config_process.get("ALIGN_WINDOW_LEFT"),config_process.get("ALIGN_WINDOW_RIGHT"))
ALIGN_MAX_SHIFT = config_process.get("ALIGN_MAX_SHIFT")


dir_set = MkDir(
    data_root=DATA_ROOT,
    key_number=KEY_TRUE,
    power_file_number=FILE_NUM,
    file_name=SOURCE_FILE_PREFIX_NAME,
    tag=DIR_TAG
)

traces_process = TraceProcess(
            sample_number=SAMPLE_NUM,
            plaintext_number=PLAINTEXT_NUM,
            save_root=SAVE_ROOT,
            save_file_name=SAVE_FILE_NAME,
            align_feature_window=ALIGN_WINDOW,
            align_max_shift=ALIGN_MAX_SHIFT
            )

if __name__ == "__main__":
    ### two modes:
    ## mkdir ; process
    ###
    parser = argparse.ArgumentParser(
        description='select mode',
        epilog='python script.py -d : 创建数据目录'
    )
    
    parser.add_argument('-d', '--mkdir', action='store_true',
                        help='创建功耗迹存储目录')
    args = parser.parse_args()
    if args.mkdir:
        dir_set.mk_dir()
    else:
        power_files = dir_set.get_power_traces_files()
        traces_process.process_traces(
            power_trace_files=power_files,
            mode=PROCESS_MODE,
            down=DOWN,
            down_num=DOWN_NUM
            )

    #print(power_files)