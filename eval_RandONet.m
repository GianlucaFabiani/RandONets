% eval_RandONet evaluates a Random projection-based Operator Network (RandONet) model by
% computing the weighted inner product between the trunk and branch networks.
%
% Syntax: G = eval_RandONet(net, ff, yy)
%
% Inputs:
%   - net : Structure containing the parameters of the RandONet model.
%           Fields include:
%             - tr_fT : Trunk network activation function (nonlinear transformation).
%             - tr_fB : Branch network activation function (nonlinear transformation).
%             - alphat, betat : Parameters for input transformation in the trunk network.
%             - alphab, betab : Parameters for input transformation in the branch network.
%             - C : Weight matrix for the inner product.
%   - ff  : Input function for the branch network.
%   - yy  : Input spatial locations for the trunk network.
%
% Output:
%   - G : Output of the RandONet model, computed as the weighted inner product
%         of the trunk and branch networks, i.e., <T, B>_C.
%
% The function transforms the inputs using the trunk and branch networks, and
% computes the result by applying the weight matrix C to the inner product of
% these transformations.
%
% DISCLAIMER: This software is provided "as is" without warranty of any kind.
% This includes, but is not limited to, warranties of merchantability,
% fitness for a particular purpose, and non-infringement.
% The authors and copyright holders are not liable for any claims, damages,
% or other liabilities arising from the use of this software.
%
% Copyright (c) 2024 Gianluca Fabiani
%
% Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
% You may not use this material for commercial purposes.
% If you remix, transform, or build upon this material,
% you must distribute your contributions under the same license as the original.

function G=eval_RandONet(net,ff,yy)
Tr=net.tr_fT(yy*net.alphat+net.betat); %trunk
Br_train=net.tr_fB(net.alphab*ff+net.betab); %branch
G=Tr*net.C*Br_train; %weighted inner product of trunk and branch <T,B>_C
end
