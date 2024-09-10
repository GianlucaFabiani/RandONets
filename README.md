# RandONet MATLAB TOOLBOX
RandONet - MATLAB Code (J. Comp. Phys).  RandONet (Random projection-based Operator Network) is a MATLAB implementation designed for learning efficiently linear and nonlinear operators using randomized neural networks.

&#x1F4D8;**Fabiani, G., Kevrekidis, I. G., Siettos, C., Yannacopoulos, A. N., RandONet: Shallow Networks with Random Projections for learning linear and nonlinear operators. J Comp Phys, (Accepted 10 sept 2024)**
arxiv at: https://doi.org/10.48550/arXiv.2406.05470

Last revised by G. Fabiani, September 11, 2024

We present a machine learning method based on ``random projections`` with ``Johnson-Lindenstrauss`` and/or ``Random Fourier Features`` for efficiently learning linear and nonlinear operators.

⭐🔍 **IMPORTANTLY**: We rigorously and theoretically prove the universal approximation of nonlinear operators with RandONets, extending the proof of Chen and Chen (1995) ✏️🔥

The efficiency of the scheme is compared against [DeepXDE/DeepOnet python library](https://github.com/lululxvi/deepxde) that implements, among others, deep-learning operator networks.

Keywords: RandONets - Machine Learning - Random Projections  - Shallow Neural Networks -  Approximation of Linear and Nonlinear Operators - Differential Equations - Evolution Operators - DeepONet - Numerical Analysis

DISCLAIMER:
This software is provided "as is" without warranty of any kind.

Abstract
=====
Deep neural networks have been extensively used for the solution of both the forward and the inverse problem for dynamical systems. However, their implementation necessitates optimizing a high-dimensional space of parameters and hyperparameters. This fact, along with the requirement of substantial computational resources, poses a barrier to achieving high numerical accuracy.
Here, to address the above challenges, we present Random Projection-based Operator Networks (RandONets): shallow networks with random projections and niche numerical analysis algorithms that learn linear and nonlinear operators. The implementation of RandONets involves: (a) incorporating random bases, thus enabling the use of shallow neural networks with a single hidden layer, where the only unknowns are the output weights of the network's weighted inner product; this reduces dramatically the dimensionality of the parameter space; and, based on this, (b) using niche numerical analysis techniques to solve a least-squares problem (e.g., Tikhonov regularization and preconditioned QR decomposition) that offer superior numerical approximation properties compared to other optimization techniques used in deep-learning.
In addition, we prove the universal approximation accuracy of RandONets for approximating linear and nonlinear operators. Furthermore, we demonstrate their efficiency in approximating linear and nonlinear evolution operators (right-hand-sides (RHS)) with a focus on PDEs. 
We show, that for this particular task, RandONets outperform both in terms of numerical approximation accuracy and computational cost, by several orders of magnitudes (~10 orders of magnitudes, up to machine precision) the ``vanilla" DeepONets. Hence, we believe that our method will trigger further developments in the field of scientific machine learning, for the development of new `'light'' machine learning schemes that will provide high accuracy while reducing dramatically the computational costs.

Matlab Examples
==========

The main function (i.e. the training) is train_RandONet.m

Here, we provide 5 examples/demos:
1) The antiderivative problem (main_RandDeepOnet_AntiDerivative.m) [as proposed in deepxde python library]
2) The pendulum  (main_RandDeepOnet_pendulum.m) [as proposed in deepxde python library]
3) Linear PDE Diffusion-Reaction (main_RandDeepOnet_DiffReac.m)
4) Nonlinear PDE Viscous Burgers' Equation (main_RandDeepOnet_burgers.m)
5) Nonlinear PDE Allen-Cahn Equation (main_RandDeepOnet_AllenCahn.m)

Description of the Problem
========
In this study, we focus on the challenging task of learning linear and nonlinear functional operators $\mathcal{F}:\mathsf{U} \rightarrow \mathsf{V}$ which constitute maps between two infinite-dimensional function spaces $\mathsf{U}$ and $\mathsf{V}$. Here, for simplicity, we consider both $\mathsf{U}$ and $\mathsf{V}$ to be subsets of the set $\mathsf{C}(\R^d)$ of continuous functions on $\R^d$. The elements of the set $\mathsf{U}$ are functions $u:\mathsf{X}\subseteq \R^d\rightarrow \R$ that are transformed to other functions $v=\mathcal{F}[u]:\mathsf{Y}\subseteq \R^d \in \R$ through the application of the operator $\mathcal{F}$. We use the following notation for an operator evaluated at a location $\bm{y} \in \mathsf{Y}\subseteq \mathbb{R}^d$
\begin{equation}
    v(\bm{y})=\mathcal{F}[u](\bm{y}).
\end{equation}
These operators play a pivotal role in various scientific and engineering applications, particularly in the context of (partial) differential equations.
By effectively learning (discovering from data) such nonlinear operators, we seek to enhance our understanding and predictive capabilities in diverse fields, ranging from fluid dynamics and materials science to financial and biological systems and beyond.

Although our objective is to learn functional operators from data, which take functions ($u$) as input, we must discretize them to effectively represent them and be able to apply network approximations. One practical approach, as implemented in the DeepONet framework, is to use the function values ($u(\bm{x}_j)$) at a sufficient, but finite, number of locations ${\bm{x}_1, \bm{x}_2, \dots , \bm{x}_m}$, where $\bm{x}_j \in \mathsf{X}\subseteq\R^d$; these locations are referred to as ``sensors."

Regarding the availability of data for the output function, we encounter two scenarios. In the first scenario, the functions in the output are known at the same fixed grid ${\bm{y}_1, \bm{y}_2,\dots,\bm{y}_{n}}$, where $y_i \in Y$; this case is termed as ``aligned" data. Conversely, there are cases where the output grid may vary randomly for each input function, known as ``unaligned" data. If this grid is uniformly sampled and dense enough, interpolation can be used to approximate the output function at fixed locations. Thus, this leads us back to the aligned data case. However, if the output is only available at sparse locations, interpolation becomes impractical. As explained in the paper, despite this challenge, our approach can address this scenario, albeit with a higher computational cost for training the machine learning model (since, in such cases, the fixed structure of the data cannot be fully leveraged).

Documentation of the Code
=====
We provide an user-friendly and MATLAB-friendly software for learning Linear and Nonlinear Operators using RandONets. The Random projection-based algorithm is a fast and efficient machine learning algorithm for function approximation.
  
**train_RandONet.m** trains a Random Projection-based Operator Network (RandONet) model.

Syntax:
net = **train_RandONet**(ff, yy, Nt, Nb, kmodel)

Inputs:
ff: Input matrix (functions) for the branch network.
yy: Input vector (spatial locations) for the trunk network.
G: Input matrix (transformed functions Gf).
Nt: Number of neurons in the trunk network (default: 200).
Nb: Number of neurons in the branch network (default: 1000).
kmodel: Model type (1 for JL, 2 for RFFN; default: 2).
Output:
net: Trained RandONet model, which contains fields for the trunk and branch networks, including weights and biases.
Structure of the net:
tr_fT: Trunk network activation function (nonlinear transformation).
tr_fB: Branch network activation function (nonlinear transformation).
alphat, betat: Parameters for input transformation in the trunk network.
alphab, betab: Parameters for input transformation in the branch network.
C: Weight matrix for the inner product.
Description:
The function initializes network parameters and trains using COD-based pseudo-inverse of the trunk and branch layers, with the results stored in the output net.