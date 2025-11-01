## Conda 环境

### 激活
`conda env acivate /15T/Projects/Dilithium-SCA/scripts/env_setting/env`
### 更新或添加某些包后将环境导出到配置文件
`conda env export > /15T/Projects/Dilithium-SCA/scripts/env_setting/python_env_conda.yml`
### 该环境只需创建一次
`conda env create --prefix /15T/Projects/Dilithium-SCA/scripts/env_setting/env -f /15T/Projects/Dilithium-SCA/scripts/env_setting/python_env_conda.yml `