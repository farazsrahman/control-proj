___

GPT Astra (or some future distillation thereof) can probably predict action chunks in real time for high-level manipulation. The natural next question is how such a LLM can be paired with fast and responsive low-level control for fine-grained manipulation. 


Some assumptions:
- LLMs will get very strong at 3D spatial reasoning
- Strong 3D LLMs will be slow and not take high-frequency feedback
- LLMs will be able to deduce a "privileged" state from only sensor images


Some motivating questions: 
- Where does Astra currently struggle in dexterous manipulation 
- What is a simple strategy to incorporate *feedback* and *high-frequency* control into Astra
- Without needing sim2real transfer, how far does such a setup takes us to general purpose robotics?


A generalizable input space for Astra is probably depth images since we can reasonbly get them from sensors & it allows us to leverage the 3d reasoning capability of Astra as opposed to one-shot image -> action chunk prediction


