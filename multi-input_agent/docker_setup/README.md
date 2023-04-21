# Instructions for TensorRT compilation



# Prerequisites

 - Linux host or Virtualized linux host that lets gpu pass through ( Tested with Ubuntu 22.04.2 LTS on Windows 11 )
 - Nvidia GPU
 - Docker on Linux host or virtualized env
 - NVIDIA Container Toolkit [User Guide — NVIDIA Cloud Native Technologies documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html)



# How to use

 1. run `launch-docker.bash` this will start the docker container if preqreqs are met, install extra depedencies, fix the buffer issue, and enter you in a bash terminal
 2. If not already done compile a model with --save-model true ie `python multi-input_dqn.py --save-model`
 3. update TRT_PATH to full path to model you want to test within the container. ie 
 `export TRT_PATH=/workspace/code/runs/current_model/multi-dqn.model`
 3. run tensorrt_compiler ie `python tensorrt_compiler.py`

## Additional analysis and benchmarking to come later
