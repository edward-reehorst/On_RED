% RED Cost vs. Fixed Point Error of "Regularization by Denoising:
% Clarifications and New Interpretations"
% https://arxiv.org/abs/1806.02296v2  
%
% An experiment to find a "bad" RED-SD trajectory
%
% A local minima is found on the RED cost surface by calculating the
% numerical gradient at every iteration and using steepest descent.
% Then, the RED-SD algorithm is initiatied at this point.
% This leads to a RED-SD trajectory where the fixed point condition is
% satisfied, but the RED Cost actually increases.
%
% Uses median filter
% 
% Edward Reehorst

%% Initialize Problem

clear;
randn('seed', 0);

% Add minimizers to path
addpath('minimizers');

input_sigma = sqrt(20);

% crop sizes
m_new = 16;
n_new = 16;

sig_f = 3.25;

%% Load image files and convert to double
i1 = double(imread('./test_images/starfish.tif'));
[m,n,~] = size(i1);

% Crop image
m_lim = floor((m-m_new)/2)+1:m-ceil((m-m_new)/2);
n_lim = floor((n-n_new)/2)+1:n-ceil((n-n_new)/2);
i1 = i1(m_lim,n_lim,:);
[m,n,o] = size(i1);

if o==3
    % If RGB convert to YCbCr and take Y channel
    i0_YCbCr = rgb2ycbcr(i1/255)*255;
    % vectorize image
    x = i0_YCbCr(:,:,1);
else
    x = i1;
end

truth = x;

% Add noise
% Noise vector
w = randn(size(x))*input_sigma;
% Blur image and add noise
y = x+w;


%% Create instant function for denoiser
% Median Filter
f = @(i,stdf)...
    medfilt2(i,[3 3]);

%% Define Regularizer
rho = @(x,fx)...
    0.5*x(:)'*(x(:)-fx(:));


%% Solve using numerical gradient
numerical_gd_iter = 1000;
ForwardFunc = @(x) x;
BackwardFunc = @(x) x;
lambda = 0.02;
mu = 1e-3;
x = y;
a_cost1 = nan(numerical_gd_iter,1);
a_psnr1 = nan(numerical_gd_iter,1);

% Define cost
cost_function = @(x,fx)...
    norm(ForwardFunc(x)-y,2)^2/(2*input_sigma^2)+lambda*0.5*x(:)'*(x(:)-fx(:));

fx = f(x,sig_f);
fprintf('Numerical Gradient Descent Progress: %3d%%',0);
for iter = 1:numerical_gd_iter
    % Size of pertubation
    p = 1e-12;
    
    grad_n = nan(m_new,n_new);
    
    costx = cost_function(x,fx);
    
    %Fill gradient/Jacobian entry by entry
    for i = 1:m_new
        for j= 1:n_new
            pert = zeros(m_new,n_new);
            pert(i,j)=p;
            fxp = f(x+pert,sig_f);

            grad_n(i,j) = (cost_function(x+pert,fxp)-costx)/p;
        end
    end
        
    % Take gradient descent step
    x = x - mu*grad_n;
    fx = f(x,sig_f);
    a_cost1(iter) = cost_function(x,fx);
    a_psnr1(iter) = psnr(truth,x,255);
    
    % Display Progress
    fprintf('\b\b\b\b%3d%%',...
        floor(iter/numerical_gd_iter*100));
end
fprintf('\n')

x_ngd = x;

%% Intialize GD at numerical gradient

% Set parameters
params = [];
params.lambda = lambda;
params.outer_iters= 100;
params.effective_sigma = sig_f;

InitEstFunc = @(y) x_ngd;

% Run SD starting from numerical gradient solution
[x_RED, a_psnr, a_cost, a_fp_error ] = RunSD_wCost(y, ForwardFunc, BackwardFunc,...
    InitEstFunc, f, input_sigma, params, truth);

%% Plot trajectories
figure;
loglog(1:numerical_gd_iter, a_cost1);
xlabel('iter')
ylabel('cost')

figure;
yyaxis left
semilogy(1:params.outer_iters, a_cost, '.-');
xlabel('iter')
ylabel('cost')

yyaxis right
semilogy(1:params.outer_iters, a_fp_error, '.-');
ylabel('fp error')
