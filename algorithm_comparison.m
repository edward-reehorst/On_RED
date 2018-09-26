% Algorithm Comparison from "Regularization by Denoising: Clarifications
% and New Interpretations" https://arxiv.org/abs/1806.02296v2  
% 
% Compares ADMM (I=1), Fixed Point, Dynamic Penalty Proximal Gradient
% (DP-PG), and Accelerated Proximal Gradient (APG), and Proximal-Gradient
% as applied to the uniform deblurring problem.
% 
% Parameters:
% ForwardFunc = 9x9 Uniform Blur operator (implemented using FFT)
% BackwardFunc = ForwardFunc
% input_sigma = sqrt(2)
% denoiser = TNRD
% sig_f = 3.25
%
% Edward Reehorst

clear;

% Set to 'DWT' for the discrete wavelet thresholding denoiser, set to
% 'TNRD' for TNRD denoiser
denoiser = 'DWT'; 

% Add minimizers to path
addpath('./minimizers/');
% Use some of the RED helper functions
addpath('./helper_functions/');

% File must be in active directory
path = './test_images/';
filename = 'starfish.tif';

% Set to True to calculate the fixed point error. Will increase the
% run-time of ADMM and APG algorithms
fp_error = true;

% Number of iterations performed of each algorithm
outer_iters = 10000;

% Create instant function for TNRD denoiser
switch denoiser
    case 'TNRD'
        % Use TNRD denoiser for comparison to RED paper
        addpath('./tnrd_denoising/');
        parpool;
        lambda = 0.02;
        effective_sigma = 3.25;
        f_denoiser = @(i,sigf) Denoiser(i,sigf);

    case 'DWT'
        %  Discrete Wavelet Domain Thresholding
        addpath('rwt/bin');
        lambda = 0.02;
        effective_sigma = 1;
        h = [1,1]/sqrt(2);
        type = 0;
        f_denoiser = @(i,sig_f)...
            denoise(i,h,type,[0,0,0,2,0,sig_f]);
end

%% read the original image
fprintf('Reading %s image...', filename);
orig_im = imread(['./test_images/'  filename]);
orig_im = double(orig_im);

fprintf(' Done.\n');


%% define the degradation model
fprintf('Test case: Uniform Blur\n');

% noise level
input_sigma = sqrt(2);
% filter size
psf_sz = 9;
% create uniform filter
psf = fspecial('average', psf_sz);
% use fft to solve a system of linear equations in closed form
use_fft = true;
% create a function-handle to blur the image
ForwardFunc = ...
    @(in_im) imfilter(in_im,psf,'conv','same','circular');
% the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
% are the same
BackwardFunc = ForwardFunc;
% special initialization (e.g. the output of other method)
% set to identity mapping
InitEstFunc = @(in_im) in_im;



%% degrade the original image


fprintf('Blurring...');
% blur each channel using the ForwardFunc
input_im = zeros( size(orig_im) );
for ch_id = 1:size(orig_im,3)
    input_im(:,:,ch_id) = ForwardFunc(orig_im(:,:,ch_id));
end
% use 'seed' = 0 to be consistent with RED experiments
randn('seed', 0);


% add noise
fprintf(' Adding noise...');
input_im = input_im + input_sigma*randn(size(input_im));

% convert to YCbCr color space if needed
input_luma_im = PrepareImage(input_im);
orig_luma_im = PrepareImage(orig_im);

fprintf(' Done.\n');
psnr_input = psnr(orig_luma_im, input_luma_im, 255);

% Default params
params = [];

% regularization factor
params.lambda = lambda;

% number of outer iterations
params.outer_iters = outer_iters;

% level of noise assumed in the regularization-denoiser
params.effective_sigma = effective_sigma;

% use fft for solving a linear system in a closed form
params.use_fft = use_fft;
params.psf = psf;

% Decide to capture fixed point error
params.fp_error = fp_error;

% number of inner iterations, set to 'nan' if use_fft == true
if use_fft
    params.inner_iters = nan;
else
    params.inner_iters = 10;
end

%% Use ADMM

fprintf('Restoring using RED: ADMM method\n');

params_admm = params;
params_admm.beta = 0.001;
params_admm.inner_denoiser_iters  = 1;
params_admm.alpha = 1;

[est_admm_im, psnr_admm, time_admm, fp_error_admm,dist_admm] = ...
                     RunADMM(input_luma_im,...
                             ForwardFunc,...
                             BackwardFunc,...
                             InitEstFunc,...
                             f_denoiser,...
                             input_sigma,...
                             params_admm,...
                             orig_luma_im);

fprintf('Done.\n');

%% Use FP

fprintf('Restoring using RED: FP method\n');

params_fp = params;

[est_fp_im, psnr_fp, time_fp, fp_error_fp,dist_fp] = ...
                       RunFP(input_luma_im,...
                             ForwardFunc,...
                             BackwardFunc,...
                             InitEstFunc,...
                             f_denoiser,...
                             input_sigma,...
                             params_fp,...
                             orig_luma_im);

fprintf('Done.\n');


%% Use PG-DP

fprintf('Restoring using RED: PG-DP method\n');

L0 = 0.2;
Lf = 2;

params_pg = params;
params_pg.Lf = Lf;
params_pg.L0 = L0;

[est_dpg_im, psnr_dpg, time_dpg,fp_error_dpg,dist_dpg] =...
                      RunDPG(input_luma_im,...
                             ForwardFunc,...
                             BackwardFunc,...
                             InitEstFunc,...
                             f_denoiser,...
                             input_sigma,...
                             params_pg,...
                             orig_luma_im);

fprintf('Done.\n');

%% APG

fprintf('Restoring using RED: APG method\n');

params_APG = params;
params_APG.L = 1;

[est_apg_im, psnr_apg, time_apg, fp_error_apg,dist_apg] =...
                      RunAPG(input_luma_im,...
                             ForwardFunc,...
                             BackwardFunc,...
                             InitEstFunc,...
                             f_denoiser,...
                             input_sigma,...
                             params_APG,...
                             orig_luma_im);

fprintf('Done.\n');


%% PG

fprintf('Restoring using RED: PG method\n');

params_PG = params;
params_PG.L = 1;

[est_pg_im, psnr_pg, time_pg,fp_error_pg,dist_pg] =...
                       RunPG(input_luma_im,...
                             ForwardFunc,...
                             BackwardFunc,...
                             InitEstFunc,...
                             f_denoiser,...
                             input_sigma,...
                             params_PG,...
                             orig_luma_im);

fprintf('Done.\n');
    
%% Analysis
fig1 = figure;
set(fig1,'renderer','Painters')
semilogx(1:outer_iters,psnr_admm,'-'...
    ,1:outer_iters,psnr_fp,'-'...
    ,1:outer_iters,psnr_dpg,'-'...
    ,1:outer_iters,psnr_apg,'-'...
    ,1:outer_iters,psnr_pg,'k--');
grid on;
xlabel('iter');
ylabel('psnr');
legend('ADMM I=1', 'FP', 'DPG','APG','PG', 'Location', 'southeast');

% normalize update distances
dist_admm = (1/size(orig_im,2)/size(orig_im,1))*dist_admm.^2;
dist_fp = (1/size(orig_im,2)/size(orig_im,1))*dist_fp.^2;
dist_dpg = (1/size(orig_im,2)/size(orig_im,1))*dist_dpg.^2;
dist_apg = (1/size(orig_im,2)/size(orig_im,1))*dist_apg.^2;
dist_pg = (1/size(orig_im,2)/size(orig_im,1))*dist_pg.^2;

fig2 = figure;
set(fig2,'renderer','Painters')
loglog(1:outer_iters,dist_admm,'-'...
    ,1:outer_iters,dist_fp,'-'...
    ,1:outer_iters,dist_dpg,'-'...
    ,1:outer_iters,dist_apg,'-'...
    ,1:outer_iters,dist_pg,'k--');
grid on;
xlabel('iter');
ylabel('normalized update dist');
legend('ADMM I=1', 'FP', 'DPG','APG','PG', 'Location', 'southwest');

if fp_error
    fig3 = figure;
    set(fig3,'renderer','Painters')
    loglog(1:outer_iters,fp_error_admm,'-'...
        ,1:outer_iters,fp_error_fp,'-'...
        ,1:outer_iters,fp_error_dpg,'-'...
        ,1:outer_iters,fp_error_apg,'-'...
        ,1:outer_iters,fp_error_pg,'k--');
    grid on;
    xlabel('iter');
    ylabel('fp error');
    legend('ADMM I=1', 'FP', 'DPG','APG','PG', 'Location', 'northeast');

end