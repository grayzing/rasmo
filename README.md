# Radio-Aware ASM Orchestrator using A2C

## Background
Substantial work has been done in finding strategies for on/off strategies for O-RUs. Since such a problem is considered to be NP-HARD, due to its combinatorial nature and mixed constraints, deep reinforcement learning shows promise in this area (Wang et. al). Previous works have used deep reinforcement learning methods to approximate a solution to this problem with success.

The issue with previous deep reinforcement learning models, however, is that they aren't easily generalizable. E.g., Wang et. al's Deep Q Learning model and Ntassah et. al's PPO model assume stationary UEs that don't otherwise enter or leave the network. Hence, these models fall flat in dynamic scenarios, e.g., urban networks, where UEs have velocity, are handed over to different O-RUs, or leave the service area.

Graph convolutional networks show promise. They are easily generalizable. As long as we have vertex features of a consistent shape (explained down below), we can use the same learnable weights. In a sense, it is one-size-fits-all.

We can model the network as a graph $G^t=(V^t,E^t)$, where edge weights $w: V \times V \to \mathbb{R}$ is defined as the RSRP from gNB $u$ to UE $v$.

$\mathbf{X} \in \mathbb{R}^{n\times6}$ is a feature vector with the geolocations of every node in the network, as well as the average instantaneous throughput of the O-RU and advanced sleep mode of the O-RU. in the case of the node being a gNB. If the node isn't a gNB, the throughput and ASM are $-1$.

$\mathbf{H} \in \mathbb{R}^{n \times n}$ is the received power matrix, which encodes the normalized RSRP between gNBs and UEs.

## Problem formulation
We want to maximize the average throughput in the system and the average received signal power while minimizing the number of active O-RUs in the network. 

The instantaneous throughput $T$ of a UE $u$ is defined as

$$
\begin{equation}
T(u)=\frac{L}{\Delta t}
\end{equation}
$$

where $L$ is the size of the packet in bits, and $\Delta t=0.001\text{s}$

The average instantaneous throughput $\mu_T$ of an O-RU $r$ with the set of connected UEs $c(r)$ is defined as

$$
\begin{equation}
\mu_T = \frac{1}{|c(r)|} \sum_{u\in c(r)} T(u)
\end{equation}
$$

The received signal power $R$ of a given UE $u$ attached to an O-RU $r$ with Tx power $P(r)$ is defined as

$$
\begin{equation}
R=P(r)-PL(u,r,f_c)
\end{equation}
$$

where $PL$ is the free-space path loss between $u$ and $r$ with frequency $f_c$. In this study, we define free-space path loss in accordance to the 3GPP UMi Street Canyon scenario.

The average received signal power $\mu_R$ of a given O-RU $r$ is defined as
$$
\begin{equation}
\mu_R = \frac{1}{|c(r)|} \sum_{u\in c(r)} R(u)
\end{equation}
$$

A sleep mode function $S: \mathcal{R} \to \mathbb{Z}$ is defined as the advanced sleep mode of a given gNB. Specifically,
$$
\begin{equation}
S(r)=\begin{cases}
0 & \text{ACTIVE} \\
1 & \text{SM1} \\
2 & \text{SM2} \\
3 & \text{SM3} \\
4 & \text{SM4} \\
\end{cases}
\end{equation}
$$

For this study, we define our objective function $y$ as
$$
\begin{equation}
y=\sum_{r\in\mathcal{R}}\mu_R(r) + \mu_T(r) + S(r)
\end{equation}
$$

To ensure adequate QoS for users, we consider a constraint on average instantaneous throughput and received signal power.

For average instantaneous throughput, we don't want it to be any less than 2 MiB/sec. For average received signal power, we don't want it to be any less than $-100$.

This becomes an optimization problem where we must maximize $y$ according to the previously defined constraints:

$$
\begin{equation}
\mathcal{P}: \max y=\sum_{r\in\mathcal{R}}\mu_R(r) + \mu_T(r) + S(r)
\end{equation}


$$
such that
$$
\mu_R(r) \geq -100 \text{dBm} \forall r\in\mathcal{R}
\newline
\mu_T(r) \geq \frac{2 \cdot 10^6 \text{bits}}{\text{second}} \forall r\in\mathcal{R}
$$

## Deep Reinforcement Learning


## Simulation
The model is validated in a UMi Street Canyon scenario with FTP type 3 traffic generation, i.e., hexagonal grid, 19 macro sites, BS antenna height 10m, 80% of UEs indoors.

O-RUs were configured with a Tx power of 35 dBm, bandwidth 100 mHz, $\mu=3$, center frequency 30 gHz, intersite distance of 200m.

eMBB traffic was modelled with a homogenous Poisson arrival process, where 4 megabit packets were transmitted downlink from O-RUs to their connected UEs at 20 times/second

The SLS was designed and implemented according to specifications in [Pedersen et. al](https://ieeexplore.ieee.org/document/10482909) and 3GPP.

Agent makes an action once every 500ms

