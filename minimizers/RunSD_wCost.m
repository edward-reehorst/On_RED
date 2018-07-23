% The SD solver used in the paper "Regularization by Denoising:
% Clarifications and New Interpretations" by Reehorst, Schniter
% https://arxiv.org/abs/1806.02296v2
%
% This code is based on RunSD that was developed as part of "The Little
% Engine That Could: Regularization by Denoising (RED)" by Romano, Elad,
% and Milanfar
% 
% Original Code from: https://github.com/google/RED
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% Performs Steepest Descent for RED. Saves cost and fixed point error
% information
%
% Cost(x) = 1/(2*input_sigma^2)||A(x)-y||^2+lambda/2*x'*(x-f(x));
%
% FP_Error(x) = ||1/input_sigma^2A'(Ax-y) + lambda(x-f(x))||^2
%
% Inputs:
%   y - the input image
%   ForwardFunc - the degradation operator H
%   BackwardFunc - the transpose of the degradation operator H
%   InitEstFunc - special initialization (e.g. the output of other method)
%   input_sigma - noise level
%   f_denosier  - Instant function for denoiser @(i,sig_f) where i is the
%   params.lambda - regularization parameter
%   params.outer_iters - number of total iterations
%   params.effective_sigma - the input noise level to the denoiser
%   orig_im - the original image, used for PSNR evaluation ONLY
%
% Outputs:
%   im_out - the reconstructed image
%   a_psnr - array of PSNR measurements between x_k and orig_im
%   a_cost - array of cost values of x_k
%   a_fp_error - array of fp_error values for x_k

function [im_out, a_psnr, a_cost, a_fp_error] = RunSD_wCost(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, f_denoiser, input_sigma, params, orig_im)

% print info every PRINT_MOD steps
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
if ~QUIET
    fprintf('%7s\t%10s\t%12s\n', 'iter', 'PSNR', 'objective');
end

% parameters
lambda = params.lambda;
outer_iters = params.outer_iters;
effective_sigma = params.effective_sigma;

cost_function = @(x,fx)...
    norm(ForwardFunc(x)-y,2)^2/(2*input_sigma^2)+lambda*0.5*x(:)'*(x(:)-fx(:));

% compute step size
mu = 0.1/(1/(input_sigma^2) + lambda);

% initialization
x_est = InitEstFunc(y);
a_psnr = nan(outer_iters,1);
a_cost = nan(outer_iters,1);
a_fp_error = nan(outer_iters,1);

f_x_est = f_denoiser(x_est, effective_sigma);
for k = 1:1:outer_iters
    
    % update the solution
    grad1 = BackwardFunc(ForwardFunc(x_est) - y)/(input_sigma^2);
    grad2 = lambda*(x_est - f_x_est);
    x_est = x_est - mu*(grad1 + grad2);
    
    % project to [0,255]
    x_est = max( min(x_est, 255), 0);
    f_x_est = f_denoiser(x_est, effective_sigma);
    
    % evaluate the cost function
    fun_val = cost_function(x_est,f_x_est);
    im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
    psnr_out = psnr(orig_im, im_out,255);
    
    a_psnr(k) = psnr_out;
    a_cost(k) = fun_val;
    a_fp_error(k) = norm(BackwardFunc(ForwardFunc(x_est)-y)/(input_sigma^2) + lambda*(x_est-f_x_est),'fro')^2;
 
    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        fprintf('%7i %12.5f %12.5f \n', k, psnr_out, fun_val);
    end
end

return

