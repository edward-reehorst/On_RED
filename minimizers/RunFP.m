% The FP solver used in the paper "Regularization by Denoising:
% Clarifications and New Interpretations" by Reehorst, Schniter
% https://arxiv.org/abs/1806.02296v2
%
% This code is modified from RunFP that was developed as part of "The Little
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

% Objective:
%   Minimize E(x) = 1/(2sigma^2)||Hx-y||_2^2 + 0.5*lambda*x'*(x-denoise(x))
%   via the fixed-point method.
%   Please refer to Section 4.2 in the paper for more details:
%   "Deploying the Denoising Engine for Solving Inverse Problems -- 
%   Fixed-Point Strategy".
%
% Inputs:
%   y - the input image
%   ForwardFunc - the degradation operator H
%   BackwardFunc - the transpose of the degradation operator H
%   InitEstFunc - special initialization (e.g. the output of other method)
%   f_denoiser  - call to denoiser, takes two inputs (in, sig_f)
%   input_sigma - noise level
%   f_denosier  - Instant function for denoiser @(i,sig_f) where i is the
%   params.lambda - regularization parameter
%   params.outer_iters - number of total iterations
%   params.inner_iters - number of inner FP iterations
%   params.use_fft - solve the linear system Az = b using FFT rather than
%                    running gradient descent for params.inner_iters. This
%                    feature is suppoterd only for deblurring
%   params.psf     - the Point Spread Function (used only when 
%                    use_fft == true).
%   params.effective_sigma - the input noise level to the denoiser
%   params.metric  - Set to 'psnr' to collect psnr data or set to 'fp' for
%                    fixed-point error.
%   orig_im - the original image, used for PSNR evaluation ONLY
%
% Outputs:
%   im_out - the reconstructed image
%   a_psnr - array of PSNR measurements between x_k and orig_im
%   a_time - array of run-times taken after the kth iteration
%   a_fp_error - array of fixed point error values after kth iteration
%   a_update_dist - array of |x_k - x_{k-1}| for kth iteration

function [im_out, a_psnr, a_time, a_fp_error,a_update_dist] = RunFP(y, ForwardFunc, BackwardFunc,...
    InitEstFunc,f_denoiser, input_sigma, params, orig_im)

% parameters
lambda = params.lambda;
outer_iters = params.outer_iters;
inner_iters = params.inner_iters;
effective_sigma = params.effective_sigma;

% print infp every PRINT_MOD steps 
QUIET = 0;
PRINT_MOD = floor(params.outer_iters/10);
if ~QUIET
    fprintf('%7s\t%10s\n', 'iter', 'psnr');
end

% initialization
tstart = tic;
x_est = InitEstFunc(y);
Ht_y = BackwardFunc(y)/(input_sigma^2);

% compute the fft of the psf (useful for deblurring)
if isfield(params,'use_fft') && params.use_fft == true
%     [h, w, ~] = size(y);
%     [l, r] = size(params.psf);
%     fft_psf = zeros(h, w);
%     t = floor((h-l)/2);
%     t2 = floor((w-r)/2);
%     fft_psf(1+t:t+l, 1+t2:t2+r) = params.psf;
%     fft_psf = fft2( fftshift(fft_psf) );
    
    [h, w, ~] = size(y);
    fft_psf = zeros(h, w);
    t = floor(size(params.psf, 1)/2);
    fft_psf(h/2+1-t:h/2+1+t, w/2+1-t:w/2+1+t) = params.psf;
    fft_psf = fft2( fftshift(fft_psf) );
    
    fft_y = fft2(y);
    fft_Ht_y = conj(fft_psf).*fft_y / (input_sigma^2);
    fft_HtH = abs(fft_psf).^2 / (input_sigma^2);
end
a_psnr = zeros(outer_iters,1);
a_fp_error = zeros(outer_iters,1);
a_time = zeros(outer_iters,1);
a_update_dist = zeros(outer_iters,1);


% apply the denoising engine
f_x_est = f_denoiser(x_est, effective_sigma);

% outer iterations
for k = 1:1:outer_iters
    x0 = x_est;
    % solve Az = b by a variant of the SD method, where
    % A = 1/(sigma^2)*H'*H + lambda*I, and
    % b = 1/(sigma^2)*H'*y + lambda*denoiser(x_est)
    if isfield(params,'use_fft') && params.use_fft == true        
        b = fft_Ht_y + lambda*fft2(f_x_est);
        A = fft_HtH + lambda;
        x_est = real(ifft2( b./A ));
        x_est = max( min(x_est, 255), 0);
    else        
        for j=1:1:inner_iters
            b = Ht_y + lambda*f_x_est;
            ht_h_x_est = BackwardFunc(ForwardFunc(x_est))/(input_sigma^2);
            res = b - ht_h_x_est - lambda*x_est;
            A_res = BackwardFunc(ForwardFunc(res))/(input_sigma^2) + lambda*res;
            mu_opt = mean(res(:).*A_res(:))/mean(A_res(:).*A_res(:));
            x_est = x_est + mu_opt*res;
            x_est = max( min(x_est, 255), 0);
        end
    end
%     if ndims(orig_im)<3
%         im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2));
%     else
%         im_out = x_est(1:size(orig_im,1), 1:size(orig_im,2), 1:size(orig_im,3));
%     end        
    im_out = x_est;
    
    a_time(k) = toc(tstart);
    
    % apply the denoising engine
    f_x_est = f_denoiser(x_est, effective_sigma);
    
    % Save current metric value
    a_psnr(k) = psnr(orig_im, im_out,255);
    a_fp_error(k) = norm(BackwardFunc(ForwardFunc(x_est)-y)/(input_sigma^2) + lambda*(x_est-f_x_est),'fro')^2;
    a_update_dist(k) = norm(x_est(:)-x0(:));

    if ~QUIET && (mod(k,PRINT_MOD) == 0 || k == outer_iters)
        % evaluate the cost function

        fprintf('%7i %12.5f \n', k, a_psnr(k));
    end
end

return

