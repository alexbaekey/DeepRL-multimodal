import torch
import torch_tensorrt
import time
import gym
from os import environ
import numpy as np
import torch.backends.cudnn as cudnn

mdqn = __import__("multi-imput_dqn")

if __name__ == "__main__":
    PATH = "/workspace/code/runs/current_model/multi-dqn.model"
    if "TRT_PATH" in environ:
        PATH = environ.get('TRT_PATH')

    run_name = f"tensorrt_comilation_{int(time.time())}"
    args = mdqn.parse_args()
    envs = gym.vector.SyncVectorEnv([mdqn.make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    
    model = mdqn.QNetwork(envs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    trt_model = torch_tensorrt.compile(model,
    inputs= [torch_tensorrt.Input((1, 8), dtype=torch.half), torch_tensorrt.Input((1, 400,600,3), dtype=torch.half)],
    enabled_precisions= {torch.half} # Run with FP16
)
