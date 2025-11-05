from re import T
from TraceProcess import TraceProcess ,MkDir

"""
    This scripts is used to process power traces of Dilithium poly module.

"""

DATA_ROOT = "/15T/Projects/Dilithium-SCA/data/traces/"
SOURCE_FILE_PREFIX_NAME = 'mau_traces-loop'

KEY_NUM = 2773
FILE_NUM =20
DIR_TAG = '_kyber'

SAMPLE_NUM = 5000
PLAINTEXT_NUM = 2994

PROCESS_MODE = 'align' # process mode: none (None) , align ('align'), denoise ('denoise'), align-denoise ('align-denoise')
DOWN = False # bool 
DOWN_NUM = 20 
SAVE_ROOT = DATA_ROOT +f'{KEY_NUM}{DIR_TAG}/averaged/'
SAVE_FILE_NAME = "averaged_mau_loop"

ALIGN_WINDOW = (330,340)
ALIGN_MAX_SHIFT = 7


dir_set = MkDir(
    data_root=DATA_ROOT,
    key_number=KEY_NUM,
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
    #mode = 'mkdir'
    mode = 'process'
    if mode == 'mkdir':
        dir_set.mk_dir()
    elif mode == 'process':
        power_files = dir_set.get_power_traces_files()
        traces_process.process_traces(
            power_trace_files=power_files,
            mode=PROCESS_MODE,
            down=DOWN,
            down_num=DOWN_NUM
            )

    #print(power_files)