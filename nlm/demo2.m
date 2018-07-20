
clc
close all;

% Noisy moon image
image = im2double(imread('moon.tif'));
noisy = imnoise(image,'gaussian',0,0.0015);

% Denoising parameters
t = 3;
f = 2;
h1 = 1;
h2 = .1;
selfsim = 0;

tic
denoised = simple_nlm(noisy,t,f,h1,h2,selfsim);
cpuTime=toc

figure(1)
subplot(2,2,1),imshow(image),title('original');
subplot(2,2,2),imshow(noisy),title('noisy');
subplot(2,2,3),imshow(denoised),title('filtered');
subplot(2,2,4),imshow(noisy-denoised),title('residuals');

mse = norm(image-denoised, 'fro')/numel(image);
psnr = 10*log10(1/mse)