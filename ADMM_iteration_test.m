% Figure 3 of "Regularization by Denoising: Clarifications and New
% Interpretations"
%
% Compares ADMM (I=1), ADMM (I=2), ADMM (I=3), and ADMM (I=4) as applied to the
% uniform deblurring problem.
%
% Parameters:
% ForwardFunc = 9x9 Uniform Blur operator (implemented using FFT)
% BackwardFunc = ForwardFunc
% input_sigma = sqrt(2)
% denoiser = TNRD
% sig_f = 3.25
% B = 0.001
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

psnr_I1 = nan(outer_iters, img_num);
psnr_I2 = nan(outer_iters, img_num);
psnr_I3 = nan(outer_iters, img_num);
psnr_I4 = nan(outer_iters, img_num);

time_I1 = nan(outer_iters, img_num);
time_I2 = nan(outer_iters, img_num);
time_I3 = nan(outer_iters,img_num);
time_I4 = nan(outer_iters, img_num);

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

    %% Use ADMM I = 1

    fprintf('Restoring using RED: ADMM I = 1 method\n');
    
    params_admm = params;
    params_admm.beta = 0.001;
    params_admm.inner_denoiser_iters  = 1;
    params_admm.alpha = 1;
    
    [~, psnr_I1(:,img_idx), time_I1(:,img_idx)] = RunADMM(input_luma_im,...
                                 ForwardFunc,...
                                 BackwardFunc,...
                                 InitEstFunc,...
                                 f_denoiser,...
                                 input_sigma,...
                                 params_admm,...
                                 orig_luma_im);

    fprintf('Done.\n');
    
    %% Use ADMM I = 2

    fprintf('Restoring using RED: ADMM I = 2 method\n');
    
    params_admm = params;
    params_admm.beta = 0.001;
    params_admm.inner_denoiser_iters  = 2;
    params_admm.alpha = 1;
    
    [~, psnr_I2(:,img_idx), time_I2(:,img_idx)] = RunADMM(input_luma_im,...
                                 ForwardFunc,...
                                 BackwardFunc,...
                                 InitEstFunc,...
                                 f_denoiser,...
                                 input_sigma,...
                                 params_admm,...
                                 orig_luma_im);

    fprintf('Done.\n');
    
    
%% Use ADMM I = 3

    fprintf('Restoring using RED: ADMM I = 3 method\n');
    
    params_admm = params;
    params_admm.beta = 0.001;
    params_admm.inner_denoiser_iters  = 3;
    params_admm.alpha = 1;
    
    [~, psnr_I3(:,img_idx), time_I3(:,img_idx)] = RunADMM(input_luma_im,...
                                 ForwardFunc,...
                                 BackwardFunc,...
                                 InitEstFunc,...
                                 f_denoiser,...
                                 input_sigma,...
                                 params_admm,...
                                 orig_luma_im);

    fprintf('Done.\n');
    
%% Use ADMM I = 4

    fprintf('Restoring using RED: ADMM I = 4 method\n');
    
    params_admm = params;
    params_admm.beta = 0.001;
    params_admm.inner_denoiser_iters  = 4;
    params_admm.alpha = 1;
    
    [~, psnr_I4(:,img_idx), time_I4(:,img_idx)] = RunADMM(input_luma_im,...
                                 ForwardFunc,...
                                 BackwardFunc,...
                                 InitEstFunc,...
                                 f_denoiser,...
                                 input_sigma,...
                                 params_admm,...
                                 orig_luma_im);

    fprintf('Done.\n');

end
%% Interpolate
% Record the minimum peak value of the time for each type of trial
time_I1_min = min(time_I1(end,:));
time_I2_min = min(time_I2(end,:));
time_I3_min = min(time_I3(end,:));
time_I4_min = min(time_I4(end,:));

grid_max = min([time_I1_min,time_I2_min,time_I3_min,time_I4_min]);

grid_min = max([time_I1(1,:), time_I2(1,:),time_I3(1,:),...
    time_I4(1,:)]);

% Create grid
grid_size = 200;
time_grid = linspace(grid_min,grid_max,grid_size);

% Declare arrays
psnr_I1_grid = zeros(grid_size,img_num);
psnr_I2_grid = zeros(grid_size,img_num);
psnr_I3_grid = zeros(grid_size,img_num);
psnr_I4_grid = zeros(grid_size,img_num);

% Interpolate
for img_idx = start_idx:end_idx
    psnr_I1_grid(:,img_idx) = interp1(time_I1(:,img_idx),psnr_I1(:,img_idx),time_grid);
    psnr_I2_grid(:,img_idx) = interp1(time_I2(:,img_idx),psnr_I2(:,img_idx),time_grid);
    psnr_I3_grid(:,img_idx) = interp1(time_I3(:,img_idx),psnr_I3(:,img_idx),time_grid);
    psnr_I4_grid(:,img_idx) = interp1(time_I4(:,img_idx),psnr_I4(:,img_idx),time_grid);
end

% Average over time steps
psnr_I1_time = mean( psnr_I1_grid, 2);
psnr_I2_time = mean( psnr_I2_grid, 2);
psnr_I3_time = mean( psnr_I3_grid, 2);
psnr_I4_time = mean( psnr_I4_grid, 2);


%% Analysis

% average psnr over each iteration
psnr_I1_iter = mean(psnr_I1,2);
psnr_I2_iter = mean(psnr_I2,2);
psnr_I3_iter = mean(psnr_I3,2);
psnr_I4_iter = mean(psnr_I4,2);


figure;
semilogx(1:outer_iters,psnr_I1_iter,'.-'...
    ,1:outer_iters,psnr_I3_iter,'.-'...
    ,1:outer_iters,psnr_I2_iter,'.-'...
    ,1:outer_iters,psnr_I4_iter,'.-');
grid on;
xlabel('iter');
ylabel('psnr (dB)');
legend('I = 1', 'I = 2', 'I = 3','I = 4', 'Location', 'southeast');

figure;
semilogx(time_grid,psnr_I1_time,'.-'...
    ,time_grid,psnr_I3_time,'.-'...
    ,time_grid,psnr_I2_time,'.-'...
    ,time_grid,psnr_I4_time,'.-');
grid on;
xlabel('time (sec)');
ylabel('psnr (dB)');
legend('I = 1', 'I = 2', 'I = 3','I = 4', 'Location', 'southeast');




