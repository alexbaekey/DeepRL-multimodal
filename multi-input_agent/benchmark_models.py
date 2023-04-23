import torch
import torch_tensorrt
import time
import gym
from os import environ
import numpy as np
import torch.backends.cudnn as cudnn

mdqn = __import__("multi-imput_dqn")


def benchmark(model, input_shape=((1, 8),(1, 400,600,3)), dtype='fp32', nwarmup=50, nruns=1000):
    input_data0 = torch.randn(input_shape[0], device='cuda')
    input_data1 = torch.randn(input_shape[1], device='cuda')   

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data0, input_data1)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc  = model(input_data0, input_data1)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    # print("Input shape:", input_data.size())
    print('Average throughput: %.2f inputs/second'%(1/np.mean(timings)))

if __name__ == "__main__":
    PATH = "/workspace/code/runs/current_model/multi-dqn.model"
    if "TRT_PATH" in environ:
        PATH = environ.get('TRT_PATH')

    run_name = f"tensorrt_comilation_{int(time.time())}"
    args = mdqn.parse_args()
    envs = gym.vector.SyncVectorEnv([mdqn.make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    
    model = mdqn.QNetwork(envs).to("cuda")
    print("\n======================Pre Optimization====================\n")
    benchmark(model, input_shape=((1, 8),(1, 400,600,3)) ,nruns=100, dtype="fp32")
    print("\n==========================================================\n\n")

    print("\n======================Pytorch Optimization====================\n")
    torch_optim_model= torch.compile(model, mode="max-autotune")
    benchmark(torch_optim_model, input_shape=((1, 8),(1, 400,600,3)) ,nruns=100, dtype="fp32")
    print("\n==========================================================\n\n")

    print("Starting model compilation (this may take some time grab a coffee)...")
    model.load_state_dict(torch.load(PATH))
    model.eval()
    trt_model = torch_tensorrt.compile(model,
    inputs= [torch_tensorrt.Input((1, 8)), torch_tensorrt.Input((1, 400,600,3))],
    enabled_precisions= { torch.half}
    # enabled_precisions= {torch.half} # Run with FP16 throwing so many issues
)
    print("Model Compiled!")

    print("\n==========================================================\n")
    print("\n======================TRT Optimization====================\n")
    benchmark(trt_model, input_shape=((1, 8),(1, 400,600,3)) ,nruns=100, dtype="fp16")
    print("\n==========================================================\n")

    torch.save(model.state_dict(), "model_state_dict")
    torch.jit.save(trt_model, "trt_QNetwork")
