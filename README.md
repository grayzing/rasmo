# GraphDQN for optimization of Energy Efficiency in 5G O-RAN

## Background
Graph structure is inherent to this problem. At time $t$, we have $K$ UEs distributed across $N$ base stations. For other models, e.g., Wang et. al, Ntassah et. al, and more, $K$ is constant across $t$, i.e., the models can't handle dynamic $K$ or $N$, thus need to be retrained for any changes.

We can model the RANs as a weighted bipartite graph $G=(V,E)$, with edges between RUs and UEs. The weights would be throughput. From here, we can use a Graph Convolutional Network to assign Q-values to each RU, and use $\text{stoch arg max}$ to get a high Q-value.

Specifically the graph convolution operator is used, where

```math
H^{l+1}=D^{\frac{-1}{2}}AD^{\frac{-1}{2}}\mathbf{H}^l\mathbf{W}^l
```

and as a special case,

```math
H^{1}=D^{\frac{-1}{2}}AD^{\frac{-1}{2}}\mathbf{X}\mathbf{W}^0
```

$\mathbf{X} \in \mathbb{R}^{(N+1+K)\times1}$ is a feature vector, where the first $N$ entries are some integer $x_i \in [1,4]$ representing what advanced sleep mode (ASM) the $i$th RU is in. $x_{N+1}$ is a dummy node, which, when put through the neural network, will be assigned a Q-value, which is the estimated reward for doing nothing at that time step. The rest of the entries, representing the UEs in the system, are zero.

Finally, to get node regressions from the final convolutional layer, we multiply $H$ by a $1\times 4$ vector to get a column vector with $N+1+K$ entries. We take the $\text{stoch arg max}$ of the first $N+1$ entries of this final vector.

## Methodology
 
![](architecture.svg)

The DQN makes use of the graph structure inherent to this problem. At time $t$, we have the bipartite graph $G^{(t)}=(V^{(t)},E^{(t)})$ representing associations between RUs and UEs. We have some weight function $w_{\text{DL}}(r_i,u_j)=B_u\log_2{(1+\zeta_u)}$, where $B_u=\aleph_uB_\text{prb}$ represents the bandwidth allocated to UE $u_j$. To meet the target throughput for UE $u_j$, we have

```math
\aleph_u=[\frac{w_\text{DL demanded}}{B_\text{prb}\log_2{(1+\zeta_u)}}]
```

where $B_\text{prb}$ is the bandwidth of one PRB. Additionally, we define the node feature vector $\mathbf{X}\in \mathbb{R}^{|V|+1\times 1}$, as mentioned in the last section.

## Simulation
The model is validated in a UMa scenario with eMBB traffic type, i.e., hexagonal grid, 19 macro sites, BS antenna height 25m, 80% of UEs indoors.

 The SLS was designed and implemented according to specifications in [Pedersen et. al](https://ieeexplore.ieee.org/document/10482909).

