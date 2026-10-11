
### September 29th, 2026

This repo is going to grow very quickly.  I am going to keep more formal notes about what I am learning  about Dexterous manipulation in the `overleaf/`, and all other maintanence / quick riffs in here. 

To start, I want to better understand the PPO algorithm. I have never really implemented it by hand, so now would be a good time to do that. At the same time it would be good for me to find a good toy RL control environment with an established reward function to iterate with.


### October 2nd, 2026

What do I want to do today?

1) Get (Claude to get) Eureka style pen spinning task running. 
2) Hack around with cart-pole (on Isaac Lab) so that I can understand impact of PPO, control space, etc. in a limited setting. 

### October 8th, 2026 

**Mani-skill Modular Jenga**:
I took some time to cook up an interesting "modular design task" inspired by Jenga. Basically a robot is given a reference design and will be. I sketched out a write up by hand in [[Design for Jenga.pdf]]. I had Claude translate this into a markdown writeup here [[Design for Jenga]]. I am going to ask Claude to implement this 



### October 11th, 2026

Last night I was riffing on a paper that I could call [[Lending Astra a Pair of Hands]] which would be written with the purpose of thinking about how to bridge from Astra to low-level dexterity. 

In order to better understand low-level dexterous manipulation training, I want to get Claude to spin up the shadow hand cube reorientation task to profile it. This will then give me an idea of how many experiments I could run and how fast it can learn. 


**NOTE:** I am probably much better off spending my time developing *evaluations* for robot policies as opposed to *policies* themselves. Some environments that I want to design / think of:
- Real time control environments
- Very general task environments (*most interesting*)
- Assembly / manufacturing environments


This should also be paired with the ability to use *flexible* embodiments. In general, I think that the "floating" end effector is a good general framework to use. 



I need to be a bit smarter about budgeting for this project as well. If I am not smart, I could end up spending a lot of personal money without need. Also, if I am smart I could get a good result by spending.
- [[Budgeting]]
- 

