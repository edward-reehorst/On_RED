clc;
close all;

% NOTE: synthetic example by Jose Vicente Manjon-Herrera

colormap(gray)

% Create synthetic image
image = 100*ones(100);
image(50:100,:) = 50;
image(:,50:100) = 2*image(:,50:100);
fs = fspecial('average');
image = imfilter(image,fs,'symmetric');

% Add some noise
sigma = 10;
noisy = image + sigma*randn(size(image));

% Denoising parameters
t = 3;
f = 2;
h1 = 1;
h2 = 10;
selfsim = 0;

tic
denoised = simple_nlm(noisy,t,f,h1,h2,selfsim);
cpuTime=toc

figure(1)
subplot(2,2,1),imagesc(image),title('original');
subplot(2,2,2),imagesc(noisy),title('noisy');
subplot(2,2,3),imagesc(denoised),title('filtered');
subplot(2,2,4),imagesc(noisy-denoised),title('residuals');

mse = norm(image-denoised, 'fro')/numel(image);
psnr = 10*log10(255^2/mse)