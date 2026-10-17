___

After chatting with Daniel from Origami robotics, he encouraged me to make a RL + low-gear-ratio motor themed research project. 


Project: How far will proprio go?

A major promise for low-gear ratio motors is that actuators proprioceptive sensing will be much more accurate without suffering from backlash (play in mesh of gear tooths when changing direction) or stiction (additional friction from transmission preventing slight contacts from being picked up or even moving the finger). There are two questions that one might ask about the low-gear ratio dexterous hand system
1) How can we quantify the benefits in modeling actuator performance through e.g. an actuator model ([[dou2026neuralactuator.pdf]],  [[hwangbo2019learning.pdf]]). Are there tradeoffs in the design? 
2) What tasks that needed tactile feedback can we actually do with proprioception (and perhaps also point clouds) alone. 
	- Cube reorientation
	- Peg-in-hole / plug insertion


Project: Modeling the tradeoffs of actuator design 



**Response to Daniel Xue**
____

Hey Daniel, 

I had a couple days to read / hack / think and came up with the following high-level proposal. I would appreciate feedback on it. Also, I am already in San Francisco and happy to come in-person to chat too. 

**Motivating Questions**: 
1) Many blind sim2real dex. manip. results use force or tactile sensors to achieve good performance \[1\]. With better sensing from QDD actuators, can we close the gap between proprioception-only and tactile/force-sensing policies? How can other actuator information such as current help \[2\]?
2) Past works have shown that actuator modeling can enable external force sensing in bi-manual manipulation \[3\] and improve sim2real RL for locomotion \[4\]. How should an actuator model be trained for a QDD hand motors, and how does accurate modeling change what is easy / hard in sim2real RL?
3) Using lessons from the above two questions, what novel tasks can we show on hardware? Perhaps piano playing (with force sensitive control of key strokes)?

**How to Get Started:**
1) Replicate blind manipulation tasks from the literature on Origami hardware (cube reorientation)
2) Build actuator models for Origami hand and ghost force sensing models to calibrate sim RL & 

**Related Work:**
- \[1\] [[yin2023rotating.pdf]] - rotating without seeing
- \[2\] [[ma2026currentastouch.pdf]] - touch sensing without touch sensing
- \[3\] [[hwangbo2019learning.pdf]] - proprioception only locomotion policies via ActuatorNets
- \[4\] [[dou2026neuralactuator.pdf]] - modeling actuators for better information
