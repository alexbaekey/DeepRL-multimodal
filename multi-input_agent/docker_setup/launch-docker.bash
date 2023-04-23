docker run -it --gpus device=1 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/../../multi-input_agent/:/workspace/code/ --rm nvcr.io/nvidia/pytorch:23.03-py3 /workspace/code/docker_setup/container-setup.sh

