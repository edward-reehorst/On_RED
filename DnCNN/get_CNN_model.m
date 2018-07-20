function [ net, modelSigma ] = get_CNN_model( noiseSigma, useGPU, useMatConvnet)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
folderModel = 'DnCNN/CNN_model';


%% load [specific] Gaussian denoising model

modelSigma  = min(75,max(10,round(noiseSigma/5)*5)); %%% model noise level
load(fullfile(folderModel,'specifics',['sigma=',num2str(modelSigma,'%02d'),'.mat']));


%%%
if useMatConvnet
    net = vl_simplenn_tidy(net);
end

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

%%% move to gpu
if useGPU && useMatConvnet
    net = vl_simplenn_move(net, 'gpu') ;
end


end

