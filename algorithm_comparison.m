% Algorithm Comparison from "Regularization by Denoising: Clarifications
% and New Interpretations" https://arxiv.org/abs/1806.02296v2  
% 
% Compares ADMM (I=1), Fixed Point, Dynamic Penalty Proximal Gradient
% (DP-PG), and Accelerated Proximal Gradient (APG) as applied to the
% uniform deblurring problem.
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

% Use TNRD denoiser for comparison to RED paper
addpath('./tnrd_denoising/');
% Add minimizers to path
addpath('./minimizers/');
% Use some of the RED helper functions
addpath('./helper_functions/');

% File must be in active directory
path = './test_images/';
filenames = {'starfish.tif', 'barbara.tif','bike.tif','boats.tif','butterfly.tif',...
    'cameraman.tif','flower.tif','girl.tif','hat.tif','house.tif','leaves.tif',...
    'lena256.tif','parrots.tif','parthenon.tif','peppers.tif',...
    'plants.tif','raccoon.tif'};
start_idx = 1;
end_idx = length(filenames);
img_num = 1 + end_idx - start_idx;

outer_iters = 200;

psnr_admm = nan(outer_iters, img_num);
psnr_dpg = nan(outer_iters, img_num);
psnr_fp = nan(outer_iters,img_num);
psnr_apg = nan(outer_iters, img_num);

time_admm = nan(outer_iters, img_num);
time_dpg = nan(outer_iters, img_num);
time_fp = nan(outer_iters,img_num);
time_apg = nan(outer_iters, img_num);

% Create instant function for TNRD denoiser
f_denoiser = @(i,sigf) Denoiser(i,sigf);

for img_idx = 1:img_num

    %% read the original image
    fprintf('Reading %s image...', filenames{img_idx+start_idx-1});
    orig_im = imread(['./test_images/'  filenames{img_idx+start_idx-1}]);
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
    % use 'seed' = 0 to be consistent with the experiments in NCSR
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
    params.lambda = 0.02;

    % number of outer iterations
    params.outer_iters = outer_iters;

    % level of noise assumed in the regularization-denoiser
    params.effective_sigma = 3.25;

    % use fft for solving a linear system in a closed form
    params.use_fft = use_fft;
    params.psf = psf;

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
    
    [est_admm_im, psnr_admm(:,img_idx), time_admm(:,img_idx)] = RunADMM(input_luma_im,...
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
    
    [est_fp_im, psnr_fp(:,img_idx), time_fp(:,img_idx)] = RunFP(input_luma_im,...
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
    
    [est_pg_im, psnr_dpg(:,img_idx), time_dpg(:,img_idx)] = RunDPG(input_luma_im,...
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

    [est_APG_im, psnr_apg(:,img_idx), time_apg(:,img_idx)] = RunAPG(input_luma_im,...
                                 ForwardFunc,...
                                 BackwardFunc,...
                                 InitEstFunc,...
                                 f_denoiser,...
                                 input_sigma,...
                                 params_APG,...
                                 orig_luma_im);

    fprintf('Done.\n');

end
%% Interpolate
% Record the minimum peak value of the time for each type of trial
grid_max = max([time_admm(end,:), time_dpg(end,:),time_fp(end,:),...
    time_apg(end,:)]);
grid_min = max([time_admm(1,:), time_dpg(1,:),time_fp(1,:),...
    time_apg(1,:)]);

% Create grid
grid_size = 200;
time_grid = linspace(grid_min,grid_max,grid_size);

% Declare arrays
psnr_admm_grid = zeros(grid_size,img_num);
psnr_dpg_grid = zeros(grid_size,img_num);
psnr_fp_grid = zeros(grid_size,img_num);
psnr_apg_grid = zeros(grid_size,img_num);

% Interpolate
for img_idx = start_idx:end_idx
    psnr_admm_grid(:,img_idx) = interp1(time_admm(:,img_idx),psnr_admm(:,img_idx),time_grid);
    psnr_dpg_grid(:,img_idx) = interp1(time_dpg(:,img_idx),psnr_dpg(:,img_idx),time_grid);
    psnr_fp_grid(:,img_idx) = interp1(time_fp(:,img_idx),psnr_fp(:,img_idx),time_grid);
    psnr_apg_grid(:,img_idx) = interp1(time_apg(:,img_idx),psnr_apg(:,img_idx),time_grid);
end

% Average over time steps
psnr_admm_time = mean( psnr_admm_grid, 2);
psnr_dpg_time = mean( psnr_dpg_grid, 2);
psnr_fp_time = mean( psnr_fp_grid, 2);
psnr_apg_time = mean( psnr_apg_grid, 2);


%% Analysis

% average psnr over each iteration
psnr_admm_iter = mean(psnr_admm,2);
psnr_dpg_iter = mean(psnr_dpg,2);
psnr_fp_iter = mean(psnr_fp,2);
psnr_apg_iter = mean(psnr_apg,2);

figure;
semilogx(1:outer_iters,psnr_admm_iter,'.-'...
    ,1:outer_iters,psnr_fp_iter,'.-'...
    ,1:outer_iters,psnr_dpg_iter,'.-'...
    ,1:outer_iters,psnr_apg_iter,'.-');
grid on;
xlabel('iter');
ylabel('psnr (dB)');
legend('ADMM I=1', 'FP', 'DPG','APG', 'Location', 'southeast');

figure;
semilogx(time_grid,psnr_admm_time,'.-'...
    ,time_grid,psnr_fp_time,'.-'...
    ,time_grid,psnr_dpg_time,'.-'...
    ,time_grid,psnr_apg_time,'.-');
grid on;
xlabel('time (sec)');
ylabel('psnr (dB)');
legend('ADMM I=1','FP', 'DPG','APG', 'Location', 'southeast');




