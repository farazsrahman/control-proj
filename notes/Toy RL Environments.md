___

I want to setup a couple of RL environments to be able to quickly test ideas. I will be selective about what I include, because the ultimate goal is that each setting should provide *some signal* on impact on real-world manipulation

**Single Task Environments**
- Shadow Dexterous Hand - https://robotics.farama.org/envs/shadow_dexterous_hand/
- Adroit Hand - https://robotics.farama.org/envs/adroit_hand/

Shadow Dexterous Hand is already ported to Isaac Lab, and I think that this is what Jason used for the original Eureka paper. What is the value in spinning this up? *I want to be able to play around with some design choices*. Specifically, I am curious about 1) Action-space 2) control-rate 3) real-time delay 4) sensory input. Evidence that any of these design choices affect performance can help me answer the questions in the affirmative:
- Does there exist a task T where design choice X impacts performance of dexterous manipulation?

A very simple place to start off with would be simply to clone the Eureka code down and get it working loca