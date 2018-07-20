% Tables 1-3 in "Regularization by Denoising: Clarifications and New
% Interpretations"
%
% An experiment to test Jacobian Symmetry, local homogeneity, and accuracy
% of RED Gradient for a variety of denoisers
%
% Edward Reehorst

%% Initialize Problem

% Denoiser Switch ('MF', 'TNRD', 'DnCNN', 'BM3D', 'NLM', 'TDT')
denoiser = 'NLM';

% File must be in active directory
path = './test_images/';
filenames = {'starfish.tif','peppers.tif','barbara.tif','bike.tif','boats.tif','butterfly.tif',...
    'cameraman.tif','flower.tif','girl.tif','hat.tif','house.tif','leaves.tif',...
    'lena256.tif','parrots.tif','parthenon.tif',...
    'plants.tif','raccoon.tif'};

p = 1e-3;  % Size of pertubation (1e-3 is best for MF?)
num_files = length(filenames);
noise_switch = 1; sig_noise = 25; sig_f = sig_noise;

% crop sizes
m_new = 16; n_new = 16;

%% Create instant function for denoiser
fprintf('Denoiser: %s\n', denoiser);
switch(denoiser)
    case 'MF'
        % Median Filter
        f = @(i,stdf)...
            reshape(medfilt2(reshape(i,[m_new,n_new]),[3 3],'symmetric'),[m_new*n_new,1]);
    case 'TNRD'
        % TNRD
        % Add TNRD folder to the path
        addpath(genpath('./tnrd_denoising/'));
        % contains basic functions
        addpath(genpath('./helper_functions/'));
        f = @(i,sig_f)...
            reshape(Denoiser(reshape(i,[m_new,n_new]),sig_f),[m_new*n_new,1]);
    case 'DnCNN'
        % DnCNN
        addpath('DnCNN');
        useGPU = true;
        use_blind = false;
        [ net, modelSigma ] = get_CNN_model( 3.25, useGPU, use_blind );
        f = @(i,sig_f)...
            reshape(CNN_denoise(reshape(i,[m_new,n_new]),net,sig_f, modelSigma, useGPU),[m_new*n_new,1]);
    case 'BM3D'
        %BM3D
        addpath('./BM3D');
        f = @(i,sig_f)...
            reshape(255*BM3D(1, reshape(i,[m_new,n_new]), sig_f),[m_new*n_new,1]);

    case 'NLM'
        %NLM
        % Scale x to not worry about setting constants
        addpath('./nlm');
        t = 3;
        s = 2;
        h1 = 1;
        h2 = .1;
        selfsim = 0;
        f = @(i,sig_f)...
            reshape(255*simple_nlm(reshape(i/255,[m_new,n_new]),t,s,h1,h2,selfsim),[m_new*n_new,1]);

    case 'TDT'
        % Transform Domain Thresholding (needs Rice Wavelet Toolbox)
        L = 2;
        h = [1,1]/sqrt(2);
        st = @(i,lam) sign(i).*max(0,abs(i)-lam);
        f = @(i,sig_f) reshape(...
            midwt( st( mdwt(reshape(i,[m_new,n_new]),h,L), sig_f), h,L),...
            [m_new*n_new,1]);
end


% Initialize Storage Arrays
a_elh1 = zeros(1,num_files);
a_elh2 = zeros(1,num_files);
a_sym_test = zeros(1,num_files);
a_grad_corr = zeros(1,num_files);
a_grad_int = zeros(1,num_files);
a_grad_romano = zeros(1,num_files);

fprintf('%3d%%',0);
for img_idx = 1:num_files

    %% Load image files and convert to double
    file1 = strcat(path,filenames{img_idx});
    i1 = double(imread(file1));
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
        x = reshape(i0_YCbCr(:,:,1),[m*n,1]);
    else
        % vectorize image
        x = reshape(i1,[m*n,1]);
    end

    truth = x;
    % Add noise, to separate x from truth (used in convex denoiser)
    if noise_switch
        % Noise vector
        w = randn(size(x))*sig_noise;
        % Blur image and add noise
        x = x+w;
    end


    %% Define Regularizer
    rho = @(x,fx)...
        0.5*x'*(x-fx);

    rhox = rho(x,f(x,sig_f));
    l = m_new*n_new;

    %% Numerical Gradient
    grad = zeros(l,1);

    Jf = zeros(m_new*n_new,m_new*n_new);
    fx = f(x,sig_f);

    %Fill gradient/Jacobian entry by entry
    for i = 1:l
        pert = zeros(l,1);
        pert(i)=p;
        fxp = f(x+pert,sig_f);
        fxm = f(x-pert,sig_f);
        grad(i) = (rho(x+pert,fxp)-rho(x-pert,fxm))/(2*p);
        Jf(:,i)=(fxp-fxm)/(2*p);
    end

    %% Other LH tests
    a_elh1(img_idx) = norm((1+p)*fx-f((1+p)*x,sig_f))^2/norm((1+p)*fx)^2;

    %% Analysis
    % Romano-claimed Gradient (needs symetric Jf and local homog)
    gradr = x-fx;

    % Correct Gradient (always true)
    gradc = x - 0.5*(fx + Jf'*x);

    % Intermediate Gradient (needs local homog)
    gradi = x - 0.5*(Jf*x + Jf'*x);

    % Local homogenaity
    a_elh2(img_idx) = norm(Jf*x-fx)^2/norm(fx)^2;

    % Test Jacobian symmetry 
    a_sym_test(img_idx) = norm(Jf-Jf.','fro')^2/norm(Jf,'fro')^2;

    % Compare numerical gradient to Romano-claimed gradient
    a_grad_romano(img_idx) = norm(gradr-grad,2)^2/norm(grad,2)^2;

    % Compare numerical gradient to correct gradient
    a_grad_corr(img_idx) = norm(gradc-grad,2)^2/norm(grad,2)^2;

    % Compare numerical gradient to intermediate gradient
    a_grad_int(img_idx) = norm(gradi-grad,2)^2/norm(grad,2)^2;

    fprintf('\b\b\b\b%3d%%',...
            floor(100*img_idx/num_files));    
end
%% Print results
fprintf('\n');

fprintf(1,'efJ:\t\t%e\n',mean(a_sym_test(:)));

fprintf(1,'(13) err:\t%e\n',mean(a_grad_romano(:)));
fprintf(1,'(38) err:\t%e\n',mean(a_grad_int(:)));
fprintf(1,'(22) err:\t%e\n',mean(a_grad_corr(:)));

fprintf(1,'eLH1:\t\t%e\n',mean(a_elh1(:)));
fprintf(1,'eLH2:\t\t%e\n',mean(a_elh2(:)));
