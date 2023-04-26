import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

dqn_path = "DQN/multi-input_agent-results/"
pytorch_opt_path = "torch-opt/pytorch-opt-results/"

dqn_sps    = pd.read_csv(os.path.join(dqn_path, "SPS.csv"))
dqnopt_sps = pd.read_csv(os.path.join(pytorch_opt_path, "SPS.csv"))

dqn_epret    = pd.read_csv(os.path.join(dqn_path, "episodic_return.csv"))
dqnopt_epret = pd.read_csv(os.path.join(pytorch_opt_path, "episode_return.csv"))

dqn_tloss    = pd.read_csv(os.path.join(dqn_path, "total_loss.csv"))
dqnopt_tloss = pd.read_csv(os.path.join(pytorch_opt_path, "td_loss.csv"))


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
plt.legend()
plt.savefig("SPS_curve.png")
plt.clf()


plt.plot(dqn_epret["Wall time"], dqn_epret["Value"], color="red", label = "MM-DQN no opt")
plt.plot(dqnopt_epret["Wall time"], dqnopt_epret["Value"], color="blue", label = "MM-DQN torch opt")
plt.title("epsiodic return")
plt.legend()
plt.savefig("epret_curve.png")
plt.clf()

plt.plot(dqn_tloss["Wall time"], dqn_tloss["Value"], color="red", label = "MM-DQN no opt")
plt.plot(dqnopt_tloss["Wall time"], dqnopt_tloss["Value"], color="blue", label = "MM-DQN torch opt")
plt.title("total loss")
plt.legend()
plt.savefig("tloss_curve.png")
plt.clf()


