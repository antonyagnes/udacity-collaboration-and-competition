# Udacity Deep Reinforcement Learning Nanodegree

# Project 2: Collaboration and Competition
In this environment, two agents control rackets to bounce a ball over a net.  The goal of each agent is to keep the ball in play.
I solved this problem using  [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) algorithm. 
### DPPG algorithm
##### Basic Actor-Critic Method
Actor-Critic agent is an agent that uses function approximation to learn the policy and the value function.
Actor and critic are two neural networks that function as follows:
**Actor** network is used as an approximate for the optimal deterministic policy. It takes the state as input and outputs distribution over the actions.
**Critic** will learn to evaluate the state value function using the TD estimate. Using this,  advantage function is be calculated. Critic takes state as its input and outputs a state value function.
#### DDPG
DDPG is a model-free kind of actor-critic (basic actor-critic method is defined earlier in this report) method. It uses a different king of actor-critic agent and it can also be called as an approximate DQN since the critic in DDPG is used to maximize over the Q values for the next state and is not used as a learned baseline.
DDPG 4 neural networks: local actor,  target actor, local critic and target critic. The important feature of this algorithm is that it uses Replay Buffers (past experiences are saved so the agent can fetch random samples  from the buffer and learn from it) and performs soft updates (updates the weights of the target networks).
### MADDPG
DDPG can be used to train multiple agents individually but there are a number of important applications that involve interaction between multiple agents, where emergent behavior and complexity arise from agents **co-evolving together**. For example, multi-robot control, the discovery of communication and language, multiplayer games, and the analysis of social dilemmas all operate in a multi-agent domain.
A general-purpose multi-agent learning algorithm proposed [here](https://arxiv.org/pdf/1706.02275.pdf) does the following: (1) leads to learned policies that only use local information (i.e. their own observations) at execution time, (2) does not assume a differentiable model of the environment dynamics or any particular structure on the communication method between agents, and (3) is applicable not only to cooperative interaction but to competitive or mixed interaction involving both physical and communicative behavior. This algorithm does centralized training with decentralized execution, allowing the policies to use extra information to ease training.
Thus, its a simple extension of actor-critic policy gradient method where the critic is augmented with extra information about the policies of other agents.
MADDPG algorithm is as follows:
![](https://lh3.googleusercontent.com/kyYJAsKqTuRNilxloTcGW6tm3WtaTMi5cZcJXDyRcrtCPflkWq8ArGcE0X-25b6F8zOCyy1Vo_o4CC6S2HQBWNyvwNsxwjcZfG02GakUrdeobuNO-ML8ppn6bboxCKEumZw6Gg0qVtwvQBVt9JAF0CKX8ZLa-xagXLAP7eD2GJRz6koco3iV-_Mx_PaqexKw-no7v5bITEXfru6Oa__YAgAU8VSwQ27JG_YuGiCrA2KKyPoSQ6qIMx20QDvjgasB3WwBAeohJgr_ACkVGhrbL299D6263uGgCJBhjRhJQ0yoXa_gQxfhMtoQRTsI5LYUQfqIVjPIGhYhu5NMKsx9PL6P-iGY9KltwfFRBHBVtrm2LhERrzp0MZLpoQJAVY0Ib2EdXxBZYhDBiqgFG2KVhouI2_bxCrXCFHSsU3sfZp1NI06g2JYUMEQ0DNVBDNNporAewzFCsW4iD5fGdpmbQB9zhKv6OF6YngtMz73s2cgoeXf1VZfjGON7JLCkCm3j1aTEy-iYRrnXlsDvNUxy1_nGF5L6-dxzQlzDZ605APCVD6UNqVh7gmXH4TU4SgcTi7Gszk_d1U3TDkJANsUx4Hfg1vxKo6dkceIXclbF-T_eQs4iyqukMTXuo-S3gbrzutkcyX7FZsTd9zZxmZvgc8siEhTW3bJIQG8uoSwt8xXNLSeLexcuxFPpzLNfLVW3qO7pTNXmmdpStdD5oYNv4ct9=w346-h268-no?authuser=0)

### Implementation
Actor and critic networks are defined inside **model.py**.
DDPG algorithm is defined inside **ddpg_agent.py**.
MADDPG algorithm is a slight modification of DDPG and it is defined inside **mddpg_agent.py**.
Follow instructions in **Tennis.ipynb** to train multi agents.
#### Hyperparameters used:
BUFFER_SIZE = 1000000 # replay buffer size

BATCH_SIZE = 1024       # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor

LR_CRITIC = 1e-3        # learning rate of the critic

WEIGHT_DECAY = 0.0   # L2 weight decay

#### Actor Network
Actor (policy) network (three fully connected layered network) maps states to actions. The first layer get the state and passes it through a hidden layer with 256 nodes(uses relu as it activation function). The output of the first layer is passed into batch normalization layer. This is then passed in to the second hidden layer with 256 nodes (uses relu as it activation function). Output of the second layer is passed into the third layer (output layer with 2 (action_size) nodes), which uses tanh has its activation function. Uses Adams optimizer.
Each agent has its own actor local and target network.  Actor networks only have  access to the information about their local agent.
#### Critic network
Critic (value) network (three fully connected layered network) that maps (state, action) pairs to Q-values.  The first layer get the state and passes it through a hidden layer with 256 nodes(uses relu as it activation function). The output of the first layer is passed into batch normalization layer. This is then passed into the second hidden layer with 256 nodes (uses relu as it activation function). Output of the second layer is passed into the third layer (output layer with 1 node), which uses tanh as its activation function. Uses Adams optimizer.  Critics networks have access to the states and actions information of both agents.
#### Batch Normalization
When learning from low dimensional feature vector observations, the different components of the observation may have different physical units (for example, positions versus velocities) and the ranges may vary across environments. This can make it difficult for the network to learn effectively and may make it difficult to find hyper-parameters which generalize across environments with different scales of state values. This issue is  addressed by adding batch normalization. This technique normalizes each dimension across the samples in a minibatch to have unit mean and variance. In addition, it maintains a running average of the mean and variance to use for normalization during testing (in our case, during exploration or evaluation).
#### Noise
A major challenge of learning in continuous action spaces is exploration. To address this issue noise class is added (Ornstein-Uhlenbeck process).
#### Replay Buffer
The  **replay buffer**  contains a collection of experience tuples from both the agents. The tuples are gradually added to the buffer.
The act of sampling a small batch of tuples from the replay buffer in order to learn. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.
### Reward Plot
The agent was able to reach a score of 0.5 at 1707th episode and it was able to maintain an average score > 0.5 for the next 100 episodes.
![enter image description here](https://lh3.googleusercontent.com/kh6KmIFTwTWTA0U3xrhwA8qqmEez-7aQ1v4hGEpJewQaBsZG3JVv85zFNyU8J2C5sDd9GF5QTwy84i0wII_qATejGN93nEOjuvq7qHz-ke2yG_4iaSCDEA-bM4AK7XrsQAmw5QTW7zAO23k_zBvZ9QDe7vCibPziueFeikhtXdoQZ9wdK5JQdem0aHWafAvHH_BZmuQ8aF4Bfn5ZMNUZscIOs5Z_CEe2Z8-Xk3Bg6Jqj3tXBBjZwVd5wSh0IsYKeMeu5LqyMsFVX0aUuFDn4CmNCEUp5Yq1qur6WKngyHola-lK2TXLQG3KYm_R2-K_ElP3HajRdCPAA3wl2Lk6sryX3WOpqY6FnfwIidLQoIdh-8cptY2S7MBzOaEPwun1jDF9wbKGFD4ego1p7XfgvRakkEVjJOUWLoS7KFJ-kI25SAv0K_QrwlX8wHKgspFLVaKquph7JTOKfGDTevcms1b7N1xyBa3rW-vlTyEWsKh1-ls0C7kn-6tRMxT4tmaedjABv8-3Du4wi1Zy4WVhiYwdqvKYMM9WNkAurj62Xohcz4NcculUDlIwKiEWymoPKONPVnAipfMjGb8sZzW7klnTj4V3h_-AC_H6aPooNERXPbPzFNtXlcVQSi-Eyh-PeH9udhTyqFvmu1MAzGtJuBsvfZCXAqLsfeCHyZSPMVkghlhkNwJaXxcvpAU6j_YJGA5Ra2-k2xBqWNjRDCav3hWWm=w990-h570-no?authuser=0)
![](https://lh3.googleusercontent.com/ucJK-MNQb90PAp1sAImoJPzvV9NeG4SoLbeo5-tBojQN2udc7QGUSxzi3NOtHul1FKgFcdR2hzkAme62o85VZeh-7ruDdmRs6-M78pD2toqYG4XVOIPJMT184srUA-TbkL7-vDncrKAcUM7Dng9XIAVYNZENuIdD4CiyKydj6UQpnh4A5PE5MDf7tV1UOJ0yEOwtBKs3fe4Ghp5Ey32VtRdO69wGOd7nHzxeByoYpoJSuqlaFsrJ1JmCl0_08KwbHJx-MHxkkUDpvtSPwhF4h_XQu6gnOysSeIZBRxW4YhB5L8f0shQJGFekshowvla3hsr3aMdmg4zJvO7Gs0KcLlwVHMWoJJyPCucSoaJyVSooT58ntcaWtWPBWnKYsicUWGgmRD2T_XnVPCZsFcDAvPj4TqYhzzTz6Y3M2G5WKwm_ruGVTWfL74oWqOdolEL2nqpuV0DDlQp4TBtsivd_5GztpWvgCTqjLFRBcl0CRx2R8eo3OpxWPvk8GLyj1gyyWmG1wXkL9TnBwHfYCGrbCEVrOSgEdGfwUFyn91sHbAX5QX3HNt3loFp-Q6-oBPGRkOILfeb48OAj6TA22A5Fv5OvQ-cpCVr_zISsek7-Lq27HeHuHCptInkobmCudfDyTKsHGNnwzOZAhR5FbPY4HtJul9XzPh39QUYIsN7rRI_7ZCGbW5RnMruezt3bm0x7IyFiSynNHE_7gDEUzMaV16gV=w1044-h602-no?authuser=0)
### Ideas for Future Work
- Use Prioritized experience replay to train the agents.
