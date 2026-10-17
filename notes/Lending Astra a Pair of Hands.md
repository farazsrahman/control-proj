___

GPT Astra (or some future distillation thereof) can probably predict action chunks in real time for high-level manipulation. The natural next question is how such a LLM can be paired with fast and responsive low-level control for fine-grained manipulation. 


Some assumptions:
- LLMs will get very strong at 3D spatial reasoning
- Strong 3D LLMs will be slow and not take high-frequency feedback
- LLMs will be able to deduce a "privileged" state from only sensor images


Some motivating questions: 
1) Where does vanilla Astra currently struggle in dexterous manipulation?
2) What is a simple strategy to incorporate *feedback* and *high-frequency* control into Astra?
3) Without needing sim2real transfer, how far does such a setup takes us to general purpose robotics?


Important to question 2) is this: how should we design a harness to allow for *general purpose* dexterous manipulation?
```sh
Minimal setup:
- Reasoning LLM Planner --> PD Controller

Rewarded Low-level:
- Reasoning LLM Planner --> Trajectory Optimization Controller
  
  
Learned Low-level:
- Reasoning LLM Planner --> RL Controller
  
  
Learned Low-level w/ Language Condition:
- Reasoning LLM Planner --> RL Controller w/ Condition on Language 


All of these can implement a simple feedback mechanism of retriggering the high-level if some large deviation from the way point is found. How to detect this is a seperate question. 
  

  
  


- Waypoints
- Language?
- Reward (for training)
- 

```




Question: 
 - Is there a general purpose language + point-cloud in -> grasp out policy out there



A generalizable input space for Astra is probably depth images since we can reasonbly get them from sensors & it allows us to leverage the 3d reasoning capability of Astra as opposed to one-shot image -> action chunk prediction





