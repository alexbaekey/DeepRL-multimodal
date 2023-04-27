import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

from math import factorial

dqn_path = "DQN/multi-input_agent-results/"
pytorch_opt_path = "torch-opt/pytorch-opt-results/"

dqn_sps    = pd.read_csv(os.path.join(dqn_path, "SPS.csv"))
dqnopt_sps = pd.read_csv(os.path.join(pytorch_opt_path, "SPS.csv"))

dqn_epret    = pd.read_csv("MMDQN-epret.csv")
dqnopt_epret = pd.read_csv("MMDQN-epret.csv")

dqn_tloss    = pd.read_csv(os.path.join(dqn_path, "total_loss.csv"))
dqnopt_tloss = pd.read_csv(os.path.join(pytorch_opt_path, "td_loss.csv"))

base_dqn    = pd.read_csv('baseDQN2.csv')

print(base_dqn)
print(dqn_epret)
def SavitzkyGolayFiltering(y, window_size, order, deriv=0, rate=1):
    #https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def walltime2sec(col):
    col["Wall time"] = col["Wall time"] - col["Wall time"][0]
    return col

walltime2sec(dqn_sps)
walltime2sec(dqnopt_sps)
walltime2sec(dqn_epret)
walltime2sec(dqnopt_epret)
walltime2sec(dqn_tloss)
walltime2sec(dqnopt_tloss)

plt.plot(dqn_sps["Wall time"], dqn_sps["Value"], color="red", label = "MM-DQN no opt")
plt.plot(dqnopt_sps["Wall time"], dqnopt_sps["Value"], color="blue", label = "MM-DQN torch opt")
plt.title("SPS")
plt.xlabel("seconds")
plt.ylabel("value")
plt.legend()
plt.savefig("SPS_curve.png")
plt.clf()


plt.plot(dqn_epret["Wall time"], dqn_epret["Value"], color="red", label = "MM-DQN no opt")
plt.plot(dqnopt_epret["Wall time"], dqnopt_epret["Value"], color="blue", label = "MM-DQN torch opt")
plt.title("epsiodic return")
plt.xlabel("seconds")
plt.ylabel("value")
plt.legend()
plt.savefig("epret_curve.png")
plt.clf()

plt.plot(dqn_tloss["Wall time"], dqn_tloss["Value"], color="red", label = "MM-DQN no opt")
plt.plot(dqnopt_tloss["Wall time"], dqnopt_tloss["Value"], color="blue", label = "MM-DQN torch opt")
plt.title("total loss")
plt.xlabel("seconds")
plt.ylabel("value")
plt.legend()
plt.savefig("tloss_curve.png")
plt.clf()


#smooth to polynomial 5
dqn_epret["Value"] = SavitzkyGolayFiltering(dqn_epret["Value"].to_numpy(), 51, 5)
base_dqn["scores"] = SavitzkyGolayFiltering(base_dqn["scores"].to_numpy(), 51, 5)


dqn_epret["Step"] = dqn_epret["Step"]/100
#print(dqn_epret)

plt.plot(dqn_epret["Step"], dqn_epret["Value"], color="red", label = "multimodal-DQN")
plt.plot(base_dqn["episode_steps"], base_dqn["scores"], color="blue", label = "vanilla DQN")
plt.title("episodic return")
plt.xlabel("episodes")
plt.ylabel("score")
plt.legend()
plt.savefig("epret_long.png")
plt.clf()


