% RED Cost Visualization from "Regularization by Denoising: Clarifications
% and New Interpretations" https://arxiv.org/abs/1806.02296v2  
%
% Visualizes the RED-cost function and negative gradients in two
% random directions
% 
% Cost(x) = 1/(2*input_sigma^2)*||x-y||^2 + lambda/2*x'(x-f(x))
% Grad(x) = 1/input_sigma^2(x-y) + lambda(x-f(x))
%
% Edward Reehorst
%

clear;

% Denoiser ('MF', 'TNRD', 'DnCNN', 'BM3D', 'NLM', 'TDT')
denoiser = 'TDT';

% Set the effective sigma (not used by all denoisers)
sig_f = 3.25;
% Set number of iterations to find RED fixed point, center of contour plot
outer_iters = 100;
% Edge length (in samples) of the grid
grid_size = 20;

addpath('minimizers');
addpath('helper_functions')


%% read the original image
filename = 'starfish.tif';
fprintf('Reading %s image...', filename);
orig_im = imread(['./test_images/'  filename]);
orig_im = double(orig_im);

fprintf(' Done.\n');


%% define denoising problem
fprintf('Test case: Denoise.\n');

% noise level
input_sigma = 10;
% filter size
psf_sz = 9;
% create identity filter filter
psf = zeros(psf_sz);
psf(ceil(psf_sz/2),ceil(psf_sz/2)) = 1;
% use fft to solve a system of linear equations in closed form
use_fft = true;
% create a function-handle to blur the image
ForwardFunc = ...
    @(in_im) in_im;
% the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
% are the same
BackwardFunc = ForwardFunc;
% special initialization (e.g. the output of other method)
% set to identity mapping
InitEstFunc = @(in_im) in_im;


%% degrade the original image

% add noise
randn('seed', 0);
fprintf(' Adding noise...');
input_im = orig_im + input_sigma*randn(size(orig_im));

% convert to YCbCr color space if needed
input_luma_im = PrepareImage(input_im);
orig_luma_im = PrepareImage(orig_im);

fprintf(' Done.\n');
psnr_input = psnr(orig_luma_im, input_luma_im, 255);

%Create instant function for denoiser
fprintf('Denoiser: %s\n',denoiser);
switch(denoiser)
    case 'MF'
        % Median Filter
        lambda = 0.00483;
        p = 30;
        f = @(i,stdf)...
            medfilt2(i,[3 3],'symmetric');
    case 'TNRD'
        % TNRD
        lambda = 0.0207;
        p = 1;
        % Add TNRD folder to the path
        addpath(genpath('./tnrd_denoising/'));
        % contains basic functions
        addpath(genpath('./helper_functions/'));
        f = @(i,sig_f)...
            Denoiser(i,sig_f);

    case 'DnCNN'
        % DnCNN
        addpath('DnCNN');
        lambda = 0.00783;
        p = 3;
        useGPU = true;
        useMatConvnet = true;
        [ net, modelSigma ] = get_CNN_model( sig_f, useGPU, useMatConvnet );
        f = @(i,sig_f)...
            CNN_denoise(i,net, useGPU, useMatConvnet);
    case 'BM3D'
        %BM3D
        lambda = 0.02069;
        p = 100;
        addpath('./BM3D');
        f = @(i,sig_f)...
           	255*BM3D(1, i, sig_f);
        
    case 'NLM'
        %NLM
        lambda = 0.00483;
        p = 10;
        % Scale x to not worry about setting constants
        addpath('./nlm');
        t = 3;
        s = 2;
        h1 = 1;
        h2 = .1;
        selfsim = 0;
        f = @(i,sig_f)...
            255*simple_nlm(i/255,t,s,h1,h2,selfsim);

    case 'TDT'
        % Transform Domain Thresholding (needs Rice Wavelet Toolbox)
        addpath('rwt/bin');
        lambda = 0.0127;
        p = 1;
        L = 2;
        h = [1,1]/sqrt(2);
        st = @(i,lam) sign(i).*max(0,abs(i)-lam);
        f = @(i,sig_f) ...
            midwt( st( mdwt(i,h,L), sig_f), h,L);
end

% define cost function
cost = @(x,fx) 0.5*(input_sigma)^-2*norm(ForwardFunc(x)-input_luma_im, 'fro')^2 + lambda/2*sum(x(:).*(x(:)-fx(:)));

% define grad function
grad = @(x,fx) (input_sigma)^-2*BackwardFunc(ForwardFunc(x)-input_luma_im)+lambda*(x-fx);


% Set parameters for FP
% Default params
params = [];

params.lambda = lambda;

% number of outer iterations
params.outer_iters = outer_iters;

% level of noise assumed in the regularization-denoiser
params.effective_sigma = sig_f;

% use fft for solving a linear system in a closed form
params.use_fft = true;
params.psf = psf;
params.inner_iters = nan;

%% USE FP to find fixed point

% Use RED minimizer to find minimum of RED cost function
[x_center, ~] = RunFP(input_luma_im,...
                                 ForwardFunc,...
                                 BackwardFunc,...
                                 InitEstFunc,...
                                 f,...
                                 input_sigma,...
                                 params,...
                                 orig_luma_im);
%% Fill grid with cost and grad values
% save size of x
xSize = size(x_center);
N = prod(xSize);

% pick two random directions
dir1 = randn(xSize)/sqrt(N);
dir2 = randn(xSize)/sqrt(N);

% create grid of test values
alpha = linspace(-p,p,grid_size);
beta = linspace(-p,p,grid_size);

% initialize storage array for cost values and gradients
a_cost_values = nan(length(alpha),length(beta));
a_grad_1 = nan(length(alpha),length(beta));
a_grad_2 = nan(length(alpha),length(beta));
fprintf('Filling Grid\n')
fprintf('Progress: %3d%%',0);
% Loop through grid to collect cost values
for a_idx = 1:length(alpha)
    for b_idx = 1:length(beta)
        x = x_center + alpha(a_idx)*dir1 + beta(b_idx)*dir2;
        fx = f(x,sig_f);
        a_cost_values(a_idx,b_idx) = cost(x,fx);
        grad_x = -grad(x,fx);
        a_grad_1(a_idx,b_idx) = sum(grad_x(:).*dir1(:));
        a_grad_2(a_idx,b_idx) = sum(grad_x(:).*dir2(:));
        
    end
    fprintf('\b\b\b\b%3d%%',...
            floor(a_idx/length(alpha)*100));    

end
fprintf('\n')

%% Contour plot
figure;
contour(beta,alpha,a_cost_values, 50);
xlabel('beta')
ylabel('alpha')
title(denoiser)
hold on
quiver(beta,alpha,a_grad_2,a_grad_1);
hold off
