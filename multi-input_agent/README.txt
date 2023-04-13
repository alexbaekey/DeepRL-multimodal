
This agent combines the view of the environment as a 600x400x3 image with a telementy vector input of size 8.

The code uses cuda GPU if you have one and CPU if not.


The environment being used is LunarLander-v2 found here
https://www.gymlibrary.dev/environments/box2d/lunar_lander/


The base agent the code is built on is CleanRL DQN (deep Q network) found here:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
DQN only uses one deep network.


To run the agent and environment use 'python multi-imput_dqn.py'.


These are th primary lib versions but different versions may work except gym[box2d] needs to be >= 0.26.  These can all be installed with conda or pip.
Torch: 	    	     1.12.1+cu102
Stable_baselines3:   1.7.0
gym[box2d]:	     0.26.2


The runs/ dir will have Tensorboard files for data analysis.

Pixel_observation.py has a wrapper class that is imported to enable multi-input.

The buffers.py file is in case you get errors from the Stable_baselines3 buffers.py file in which case you will need to replace it with the one in this dir.


Enjoy!

