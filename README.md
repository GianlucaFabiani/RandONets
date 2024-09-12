# RandONet MATLAB TOOLBOX
RandONet - MATLAB Code (J. Comp. Phys).  RandONet (Random projection-based Operator Network) is a MATLAB implementation designed for learning efficiently linear and nonlinear operators using randomized neural networks.

&#x1F4D8;**Fabiani, G., Kevrekidis, I. G., Siettos, C., Yannacopoulos, A. N., RandONets: Shallow Networks with Random Projections for learning linear and nonlinear operators. J Comp Phys, (Accepted 10 sept 2024)**
arxiv at: https://doi.org/10.48550/arXiv.2406.05470

Last revised by G. Fabiani, September 12, 2024

<img src="https://raw.githubusercontent.com/GianlucaFabiani/RandONets/main/images/Schematic_RandOnet_details_colored.jpg" alt="Schematic of RandOnet" width="600"/>

We present a machine learning method based on ``random projections`` with ``Johnson-Lindenstrauss (JL)`` and/or Rahimi and Recht (2007) ``Random Fourier Features (RFFN)`` for efficiently learning linear and nonlinear operators.

⭐🔍 **IMPORTANTLY**: We rigorously and theoretically prove the universal approximation of nonlinear operators with RandONets, extending the proof of Chen and Chen (1995) ✏️🔥

The efficiency of the scheme is compared against [DeepXDE/DeepOnet python library](https://github.com/lululxvi/deepxde) that implements, among others, deep-learning operator networks.

Keywords: RandONets - Machine Learning - Random Projections  - Shallow Neural Networks -  Approximation of Linear and Nonlinear Operators - Differential Equations - Evolution Operators - DeepONet - Numerical Analysis

**DISCLAIMER**:
This software is provided "as is" without warranty of any kind., without any express or implied warranties.
This includes, but is not limited to, warranties of merchantability, fitness for a particular purpose, and non-infringement.
The authors and copyright holders are not liable for any claims, damages, or other liabilities arising from the use of this software

Copyright (c) 2024 Gianluca Fabiani

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You may not use this material for commercial purposes.
If you remix, transform, or build upon this material, you must distribute your contributions under the same license as the original.

Abstract
=====
Deep neural networks have been extensively used for the solution of both the forward and the inverse problem for dynamical systems. However, their implementation necessitates optimizing a high-dimensional space of parameters and hyperparameters. This fact, along with the requirement of substantial computational resources, pose a barrier to achieving high numerical accuracy, but also interpretability.
Here, to address the above challenges, we present Random Projection-based Operator Networks (RandONets): shallow networks with random projections and tailor-made numerical analysis algorithms that learn linear and nonlinear operators. The implementation of RandONets involves: (a) incorporating random bases, thus enabling the use of shallow neural networks with a single hidden layer, where the only unknowns are the output weights of the network's weighted inner product; this reduces dramatically the dimensionality of the parameter space; and, based on this, (b) using tailor-made numerical analysis techniques to solve a linear ill-posed problem with regularization (e.g., Tikhonov regularization and preconditioned QR decomposition). 
In addition, we prove the universal approximation accuracy of RandONets for approximating linear and nonlinear operators. Furthermore, we demonstrate their efficiency in approximating linear and nonlinear evolution operators (right-hand-sides (RHS)) with a focus on PDEs. We also note that due to their simplicity, RandONets provide a one-step transformation of the input space, facilitating the interpretability.
We show, that for this particular task, RandONets outperform both in terms of numerical approximation accuracy and computational cost, by several orders of magnitudes the ``vanilla" DeepONets. Hence, we believe that our method will trigger further developments in the field of scientific machine learning, for the development of new `'light'' machine learning schemes that will provide high accuracy while reducing dramatically the computational costs.

Matlab Examples
==========

The main function (i.e. the training) is train_RandONet.m

Here, we provide 5 examples/demos in the file main_RandDeepOnet_examples.m:

1) The antiderivative problem (load('data_antiderivative.mat')) [as proposed in deepxde python library]
2) The pendulum with external force  (load('data_Pendulum.mat')) [as proposed in deepxde python library]
3) Linear PDE Diffusion-Reaction (load('data_DiffReac.mat'))
4) Nonlinear PDE Viscous Burgers' equation (load('data_burgers.mat'))
5) Nonlinear PDE Allen-Cahn equation (load('data_AllenCahn.mat'))

Description of the Problem
========
In this study, we focus on the challenging task of learning linear and nonlinear functional operators $\mathcal{F}:\mathsf{U} \rightarrow \mathsf{V}$ which constitute maps between two infinite-dimensional function spaces $\mathsf{U}$ and $\mathsf{V}$. Here, for simplicity, we consider both $\mathsf{U}$ and $\mathsf{V}$ to be subsets of the set $\mathsf{C}(\mathbb{R}^d)$ of continuous functions on $\mathbb{R}^d$. The elements of the set $\mathsf{U}$ are functions $u:\mathsf{X}\subseteq \mathbb{R}^d \rightarrow \mathbb{R}$ that are transformed to other functions $v=\mathcal{F}[u]:\mathsf{Y}\subseteq \mathbb{R}^d \in \mathbb{R}$ through the application of the operator $\mathcal{F}$. We use the following notation for an operator evaluated at a location $y \in \mathsf{Y}\subseteq \mathbb{R}^d$

$v(y)=\mathcal{F}[u] (y).$

These operators play a pivotal role in various scientific and engineering applications, particularly in the context of (partial) differential equations.
By effectively learning (discovering from data) such nonlinear operators, we seek to enhance our understanding and predictive capabilities in diverse fields, ranging from fluid dynamics and materials science to financial and biological systems and beyond.

Although our objective is to learn functional operators from data, which take functions ($u$) as input, we must discretize them to effectively represent them and be able to apply network approximations. One practical approach, as implemented in the DeepONet framework, is to use the function values ($u(x_j)$) at a sufficient, but finite, number of locations ${x_1, x_2, \dots , x_m}$, where $x_j \in \mathsf{X}\subseteq\mathbb{R}^d $; these locations are referred to as ``sensors."

Regarding the availability of data for the output function, we encounter two scenarios. In the first scenario, the functions in the output are known at the same fixed grid ${y_1, y_2,\dots,y_{n}}$, where $y_i \in Y$; this case is termed as "aligned" data. Conversely, there are cases where the output grid may vary randomly for each input function, known as "unaligned" data. If this grid is uniformly sampled and dense enough, interpolation can be used to approximate the output function at fixed locations. Thus, this leads us back to the aligned data case. However, if the output is only available at sparse locations, interpolation becomes impractical. As explained in the paper, despite this challenge, our approach can address this scenario, albeit with a higher computational cost for training the machine learning model (since, in such cases, the fixed structure of the data cannot be fully leveraged).

Documentation of the Code
=====
We provide an user-friendly and MATLAB-friendly software for learning Linear and Nonlinear Operators using RandONets. The Random projection-based algorithm is a fast and efficient machine learning algorithm for function approximation.
  
**train_RandONet.m** trains a Random Projection-based Operator Network (RandONet) model.

Syntax:
net = **train_RandONet**(ff, yy, Nt, Nb, kmodel)

Inputs:

* ff: Input matrix (functions) for the branch network.
* yy: Input vector (spatial locations) for the trunk network.
* G: Input matrix (transformed functions G(ff) ).
* Nt: Number of neurons in the trunk network (default: 200).
* Nb: Number of neurons in the branch network (default: 1000).
* kmodel: Model type (1 for JL, 2 for RFFN; default: 2).

Output:

* net: Trained RandONet model, which contains fields for the trunk and branch networks, including weights and biases.

Structure of the net:

* tr_fT: Trunk network activation function (nonlinear transformation).
* tr_fB: Branch network activation function (nonlinear transformation).
* alphat, betat: Parameters for input transformation in the trunk network.
* alphab, betab: Parameters for input transformation in the branch network.
* C: Weight matrix for the inner product.

**Description**:

The function initializes network parameters and trains using COD-based pseudo-inverse of the trunk and branch layers, with the results stored in the output net.

--------

**eval_RandONet** evaluates a Random projection-based Operator Network (RandONet) model by computing the weighted inner product between the trunk and branch networks.

Syntax: G = **eval_RandONet**(net, ff, yy)

Inputs:
  * net : Structure containing the parameters of the RandONet model.

Fields include:
    - tr_fT : Trunk network activation function (nonlinear transformation).
    - tr_fB : Branch network activation function (nonlinear transformation).
    - alphat, betat : Parameters for input transformation in the trunk network.
    - alphab, betab : Parameters for input transformation in the branch network.
    - C : Weight matrix for the inner product.
    - ff  : Input function for the branch network.
    - yy  : Input spatial locations for the trunk network.

Output:
  - G : Output of the RandONet model, computed as the weighted inner product of the trunk and branch networks, i.e., <T, B>_C.

The function transforms the inputs using the trunk and branch networks, and computes the result by applying the weight matrix C to the inner product of these transformations.


