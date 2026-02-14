# Radio-Aware ASM Orchestrator using A2C

## Background
Graph structure is inherent to this problem. At time $t$, we have $K$ UEs distributed across $N$ base stations. For other models, e.g., Wang et. al, Ntassah et. al, and more, $K$ is constant across $t$, i.e., the models can't handle dynamic $K$ or $N$, thus need to be retrained for any changes. This model is radio-aware and works with dynamic $K$ and $N$, so it's ideal for, say, mobile devices in an urban scenario.

We can model the network as a graph $G^t=(V^t,E^t)$, where edge weights $w: V \times V \to \mathbb{R}$ is defined as the RSRP from gNB $u$ to UE $v$.

$\mathbf{X} \in \mathbb{R}^{n\times6}$ is a feature vector with the geolocations of every node in the network, as well as the average throughput of connected UEs, advanced sleep mode of the gNB, and percentage of PRB utilization, in the case of the node being a gNB. If the node isn't a gNB, the throughput, ASM, and PRB utilization are $-1$.

$\mathbf{H} \in \mathbb{R}^{n \times n}$ is the received power matrix, which encodes the normalized RSRP between gNBs and UEs.

## Problem formulation
We want to maximize the average throughput in the system as well as minimize the power usage in the system. A sleep mode function $S: \mathcal{R} \to \mathbb{Z}$ is defined as the advanced sleep mode of a given gNB. Average throughput is defined as 


## Simulation
The model is validated in a UMa scenario with eMBB traffic type, i.e., hexagonal grid, 19 macro sites, BS antenna height 25m, 80% of UEs indoors.

The SLS was designed and implemented according to specifications in [Pedersen et. al](https://ieeexplore.ieee.org/document/10482909) and 3GPP.

Agent makes an action once every 500ms

