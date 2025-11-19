ENV_PATH := /15T/Projects/Dilithium-SCA/scripts/env_setting/envs
ENV_NAME := dilithium-sca
INTEPRETER := $(ENV_PATH)/$(ENV_NAME)/bin/python3

ENV_FILE := python_env_conda.yml

# 默认目标
.DEFAULT_GOAL := help

# 帮助信息
help:
	@echo "可用命令:"
	@echo "  make cpa      运行CPA攻击"
	@echo "  make process  处理功耗迹曲线"
	@echo "  make dir      自动生成数据目录"
	@echo "  Tips: 修改配置文件 setting.json 中的 KEY_TRUE、FILE_NUM、DIR_TAG 等属性后 make dir 创建存放功耗迹的目录 "
	@echo "  Tips: 修改配置文件 setting.json 中的 REMARK 属性在 <结果目录文件夹名> 记录此次修改 "



cpa: DilithiumCPA.py CpaAttack.py
	@$(INTEPRETER) DilithiumCPA.py

process: DilithiumTraces.py TraceProcess.py
	@$(INTEPRETER) DilithiumTraces.py

dir: DilithiumTraces.py TraceProcess.py 
	@$(INTEPRETER) DilithiumTraces.py -d
