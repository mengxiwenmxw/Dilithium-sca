# ************************** Project Information *************************** #
# Project     : Dilithium                  			                         #
# Author      : Yiming Chen, Meng Xiwen                                      #
# Date        : 2025-06-07                                                   #
# Toolchain   : Verilator + C++ Testbench + GTKWave; VCS + Verdi             #
# ************************************************************************** #

########################
#      Module Name     #
########################
MODULE := POLY_MAU
# MODULE := lbus_if
# MODULE := chip_sasebo_giii_poly
# MODULE := test_power

#########################
#     Path & Source     #
#########################
WORK_DIR := $(shell pwd)

RTL_DIR := $(WORK_DIR)/../rtl
TB_DIR := $(WORK_DIR)/../tb
BIN_DIR := $(WORK_DIR)/bin
LOG_DIR := $(WORK_DIR)/logs
WAVE_DIR := $(WORK_DIR)/waves

##############
#    SRCS    #
##############
TOP_SRCS := $(wildcard $(RTL_DIR)/$(MODULE).v)

RTL_SRCS := $(wildcard $(RTL_DIR)/*.v)

TB_SRCS_CPP := $(wildcard $(TB_DIR)/tb_$(MODULE).cpp)
TB_SRCS_V := $(wildcard $(TB_DIR)/tb_$(MODULE).v)
#$(info MODULE = $(MODULE))
#$(info RTL_DIR = $(RTL_DIR))
#$(info Looking for: $(RTL_DIR)/$(MODULE).v)
#$(info TOP_SRCS = $(TOP_SRCS))

##########################
#     VCS Simulation     #
##########################
VCS_FLAGS := 	-full64 \
				-sverilog \
				-debug_access+all \
				+fsdb+memory \

INCLUDE_FLAGS := +incdir+$(RTL_DIR)

.PHONY:vcs_compile
vcs_compile:
	@echo
	@echo ">>> VCS Compiling >>>"
	@echo $(TOP_SRCS)
	mkdir -p $(WAVE_DIR) $(BIN_DIR) $(LOG_DIR)
	vcs $(VCS_FLAGS) $(INCLUDE_FLAGS) $(TB_SRCS_V) $(RTL_SRCS) -l $(LOG_DIR)/compile.log -o $(BIN_DIR)/sim -Mdir=$(BIN_DIR)/csrc

.PHONY:vcs_sim
vcs_sim: vcs_compile
	$(BIN_DIR)/sim -l $(LOG_DIR)/sim.log +notimingcheck +nospecify -k ucli.key

.PHONY:verdi
verdi: vcs_sim
	verdi -sv -ssf $(WAVE_DIR)/tb_$(MODULE).fsdb

# ===== detect the problems without compiling ===== #
.PHONY:lint
lint: $(TOP_SRCS)
	verilator --lint-only $(TOP_SRCS)

.PHONY: clean
clean:
	rm -rf \
		.stamp.* \
		./obj_dir \
		$(BIN_DIR) \
		$(LOG_DIR) \
		ucli.key \
		verdiLog \
		novas_dump.log \
		novas.conf \
		novas.rc