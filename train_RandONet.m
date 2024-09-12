% train_RandONet trains a Random Projection-based Operator Network (RandONet) model.
%
% Syntax: net = train_RandONet(ff, yy, Nt, Nb, kmodel)
%
% Inputs:
%   - ff      : Input matrix (functions) for the branch network.
%   - yy      : Input vector (spatial locations) for the trunk network.
%   - G       : Input matrix (transformed functions G[f](y)
%   - Nt      : Number of neurons in the trunk network (default: 200).
%   - Nb      : Number of neurons in the branch network (default: 1000).
%   - kmodel  : Model type (1 for JL, 2 for RFFN; default: 2).
%
% Output:
%   - net     : Trained RandONet model containing fields for the trunk and
%               branch networks, including weights and biases.
%
%   - net : Structure containing the parameters of the RandONet model.
%           Fields include:
%             - tr_fT : Trunk network activation function (nonlinear transformation).
%             - tr_fB : Branch network activation function (nonlinear transformation).
%             - alphat, betat : Parameters for input transformation in the trunk network.
%             - alphab, betab : Parameters for input transformation in the branch network.
%             - C : Weight matrix for the inner product.
%
% The function initializes network parameters, trains using COD-based pseudo-inverse 
% of the trunk and branch layers, and stores the results in the output net.
%
% DISCLAIMER: This software is provided "as is" without warranty of any kind.
% This includes, but is not limited to, warranties of merchantability,
% fitness for a particular purpose, and non-infringement.
% The authors and copyright holders are not liable for any claims, damages,
% or other liabilities arising from the use of this software.
%
%Copyright (c) 2024 Gianluca Fabiani
%
%Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
% You may not use this material for commercial purposes.
% If you remix, transform, or build upon this material,
% you must distribute your contributions under the same license as the original.

function net=train_RandONet(ff,yy,G,Nt,Nb,kmodel)
%check arguments %basic selection
if nargin==3
    Nt=200;
    Nb=1000;
    kmodel=2; %RFFN
end
if nargin==4
    Nb=1000;
    kmodel=2; %RFFN
end
if nargin==5
    kmodel=2; %RFFN
end
if isempty(Nt)
    Nt=200;
end
if isempty(Nb)
    Nb=1000;
end
if isempty(kmodel)
    kmodel=2;
end
%
%main part of training
tr_fT=@(x) tanh(x); %activation function of the trunk network (only this one tested)
if kmodel==1
    tr_fB=@(x) x; flag_branch=1; %%%%%%%JL based model
elseif kmodel==2
    tr_fB=@(x) cos(x); flag_branch=0;  %%%%%%RFFN based model
end
Nx=size(ff,1); %number of discretization points of functions
Ny=size(yy,1); %number of points of the evaluation grid
%
x0=min(yy); xf=max(yy); %spatial interval
DX=xf-x0; %size of the interval
alphat=2*(4+Nt*9/100)*(2*rand(1,Nt)-1)/DX; %random internal weights of the trunk
cent=linspace(x0,xf,Nt);
betat=-alphat.*cent; % random biases of the trunk
Tr=tr_fT(yy*alphat+betat); %trunk hidden layer
%branch net
U0=min(min(ff,[],2));
Uf=max(max(ff,[],2));
DU=Uf-U0; %normalization of the function space
U0=U0-DU/10;
Uf=Uf+DU/10;
if flag_branch==1 %JL
    alphab=sqrt(1/Nb)*randn(Nb,Nx)/DU; %JL random internal weighrs in the branch
    betab=zeros(Nb,1);
end
if flag_branch==0
    alphab=1/Nx*sqrt(2/Nb)*randn(Nb,Nx)/DU; %RFFN random internal weighrs in the branch
    betab=rand(Nb,1)*2*pi;
end
%         if flag_branch==2 (not tested)
%             alphab=1/Nx*(4+Nb.^(1/Nx)*9/100)*(2*rand(Nb,Nx)-1)./DU;
%             cenb=U0+DU.*rand(Nb,Nx);
%             betab=-sum(alphab.*cenb,2);
%         end
Br_train=tr_fB(alphab*ff+betab); %branch hidden layer
[Qr,T11,Vp]=pinvCOD(Tr); %COD based pseudo inverse of the trunk hidden layer
[Qr2,T112,Vp2]=pinvCOD(Br_train,[]); %COD based pseudo inverse of the branch hidden layer
%
C=((Vp*(T11\(Qr'*G))*Vp2)/T112)*Qr2'; %external weights
%save net
net.tr_fT=tr_fT;
net.alphat=alphat;
net.betat=betat;
net.tr_fB=tr_fB;
net.alphab=alphab;
net.betab=betab;
net.C=C;
end