# Data-driven Inventory Management Modeling using Deep Reinforcement Learning

Reinforcement Learning is getting increasingly relevant in the field of Business, especially 
with the explosive growth of IoT and autonomous technologies that were promoted by the Fourth
Industrial Revolution. In this repository, a data-driven Make-to-Stock (MTS) Inventory Management 
is conducted with Deep Reinforcement Learning Agents. The agent learns on a sequential decision-making
basis to optimize the production and allocation of apparel products. It learns with the objective 
to establish the right balance between appropriate storage and demand fulfillment. 

The environment is a two-echelon Inventory Management system with 6 retail shops.

<div align="center">
<img src="https://github.com/user-attachments/assets/b9870498-fad2-491b-a769-05bf9a253574" width=80% height=80%>
</div><br />

In Reinforcement Learning, reward function plays an essential role for the learning process and optimization
of the policy. In this work, a novel reward function named Total Penalty (TP), was proposed and used. 

Convergence of the original reward function (Total Cost) of Random Policy, A2C, TRPO, TD3, and SAC

<div align="center">
<img src="https://github.com/user-attachments/assets/8696d58b-72f4-4d56-bf89-4fd521c2b1ca" width=70% height=70%>
</div><br />

With the TP reward function, the agent conducted the inventory management with the lowest Total Cost 
and inventory-to-sales ratio. Particularly, Soft Actor Critic (SAC) agent was able to achieve such feats 
by leveraging on its entropy maximization technique and be robust to highly unstable demand. This is further
backed up by the eventual lowest convergence of its TP Reward function's plot. 

<div align="center">
<img src="https://github.com/user-attachments/assets/204b8b21-e7ef-4b5a-84ba-272714829398" width=70% height=70%>
</div><br />
