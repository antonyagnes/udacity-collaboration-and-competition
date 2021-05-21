
# Udacity Deep Reinforcement Learning Nanodegree

# Project 3: Collaboration and Competition
![](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)
## Goal
In this environment, two agents control rackets to bounce a ball over a net.  The goal of each agent is to keep the ball in play.

## Environment
The environment provided by [Udacity](www.udacity.com) is similar to the one built by [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) .

### State space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. 

### Action space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
### Reward
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.

### Expected result
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

-   After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
-   This yields a single  **score**  for each episode.

The environment is considered solved, when the average (over 100 episodes) of those  **scores**  is at least +0.5.

## Set up Instructions

If you haven't already, please follow the  [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)  to set up your Python environment. These instructions can be found in  `README.md`  at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(_For Windows users_) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Getting Started

For this project, you will  **not**  need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

-   Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
-   Mac OSX:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
-   Windows (32-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
-   Windows (64-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the  `p3_collab-compet/`  folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(_For Windows users_) Check out  [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)  if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not  [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use  [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)  to obtain the "headless" version of the environment. You will  **not**  be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (_To watch the agent, you should follow the instructions to  [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the  **Linux**  operating system above._)

### Included in this repository
-   The code used to create and train the Agent
    -   Tennis.ipynb
    -   dppg_agent.py
    -   maddpg_agent.py
    -   model.py 
-   The trained model
    -   checkpoint_actor_0.pth (actor for the first agent).
    -   checkpoint_actor_1.pth (actor for the second agent).
    -   checkpoint_critic_0.pth (critic for the first agent).
    -   checkpoint_critic_1.pth (critic for the second agent).
-   A Report.pdf file describing the development process and the learning algorithm, along with ideas for future work

### Instructions to run the code
Open  `Tennis.ipynb` and follow the instructions in the notebook.

