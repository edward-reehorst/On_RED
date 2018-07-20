% This code downloaded from the website of Yunjin Chen
% http://www.icg.tugraz.at/Members/Chenyunjin/about-yunjin-chen
%
% This is an implementation of the paper by Yunjin Chen, and Thomas Pock, 
% "Trainable Nonlinear Reaction Diffusion: A Flexible Framework for Fast 
% and Effective Image Restoration", IEEE TPAMI 2016.
%
% When downloading the package, one can find in the following path
% ./TNRD-Codes/TestCodes(denoising-deblocking-SR)/GaussianDenoising/
% the function DenoisingOneImg.m, which cleans an example image.
% This function - ReactionDiffusion.m - is based on DenoisingOneImg.m, 
% modified to handle a given input image rather than a fixed one.

function recover = ReactionDiffusion(Im)

% clc;
% load JointTraining_7x7_400_180x180_stage=5.mat;
load JointTraining_7x7_400_180x180_stage=5_sigma=5.mat;

filter_size = 7;
m = filter_size^2 - 1;
filter_num = 48;
BASIS = gen_dct2(filter_size);
BASIS = BASIS(:,2:end);
%% pad and crop operation
bsz = 8;
bndry = [bsz,bsz];
pad   = @(x) padarray(x,bndry,'symmetric','both');
crop  = @(x) x(1+bndry(1):end-bndry(1),1+bndry(2):end-bndry(2));
%% MFs means and precisions
KernelPara.fsz = filter_size;
KernelPara.filtN = filter_num;
KernelPara.basis = BASIS;
%% MFs means and precisions
trained_model = save_trained_model(cof, MFS, stage, KernelPara);

[R,C] = size(Im);

%% run denoising, 5 stages
input = pad(Im);
noisy = pad(Im);
run_stage = 5;
for s = 1:run_stage
    deImg = denoisingOneStepGMixMFs(noisy, input, trained_model{s});
    t = crop(deImg);
    deImg = pad(t);
    input = deImg;
end
x_star = t(:);
%% recover image
recover = reshape(x_star,R,C);

