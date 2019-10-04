# Machine learning/deep learning in molecular dynamics

A repository of update in molecular dynamics field by recent progress in machine learning and deep learning. Those efforts are cast into the following categories: 
1. Learn the force field or molecular interactions;  
2. Enhanced sampling methods;
3. Learn the collective variable to bias enhanced sampling;  
4. Capture the dynamics of the molecular system; 
5. Map between all atoms and coarse grain;


&nbsp;  

<img src="https://pubs.rsc.org/en/Content/Image/GA/C7SC02267K" align="center" alt="Machine learning molecular dynamics for the simulation of infrared spectra">
(Picture from Machine learning molecular dynamics for the simulation of infrared spectra. )
&nbsp;  




### 4. Capture the dynamics of the molecular system 

[Equivariant Hamiltonian Flows](https://arxiv.org/abs/1909.13739)   
Danilo Jimenez Rezende, Sébastien Racanière, Irina Higgins, Peter Toth.  
This paper from Google uses Lie algebra to prove what hamiltonian flow learns and how addition of symmetry invariance constraints can improve data efficiency. 

[Symplectic ODE-NET: learning Hamiltonian dynamics with control](https://arxiv.org/abs/1909.12077)    
Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty.    
This paper from Princeton University and Siemens Corp infers the dynamics of a physical system from observed state trajectories. They embedded high dimensional coordinates into low dimensions and velocity into general momentum. 

[Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563)   
Sam Greydanus, Misko Dzamba, Jason Yosinski.    
This paper from Google, PetCube and Uber trains models to learn conservation law of Hamiltonian in unsupervised way.  


[Symplectic Recurrent Neural Networks](https://arxiv.org/abs/1909.13334)   
Zhengdao Chen, Jianyu Zhang, Martin Arjovsky, Léon Bottou.   
The authors from NYU, Tianjin University, and Facebook proposes SRNN to capture the dynamics of physical systems from observed trajectories.  

[Physical Symmetries Embedded in Neural Networks](https://arxiv.org/abs/1904.08991)   
M. Mattheakis, P. Protopapas, D. Sondak, M. Di Giovanni, E. Kaxiras.   
The authors from Harvard and Polytechnic Milan used symplectic neural network to embed physics symmetry in the neural network to characterize the dynamics. 

