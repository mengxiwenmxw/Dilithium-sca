## Conda 环境

### 激活
`conda env acivate dilithium-sca`
### 更新或添加某些包后将环境导出到配置文件
`conda env export > /15T/Projects/Dilithium-SCA/scripts/env_setting/python_env_conda.yml`
### 该环境只需创建一次
`./conda_create_envs `
### 环境如果已经存在但在自己的conda env list下无法查看，尝试:
`conda config --add env_dirs /15T/Projects/Dilithium-SCA/scripts/env_setting/envs`