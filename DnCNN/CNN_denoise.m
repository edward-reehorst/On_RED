function [ output ] = CNN_denoise( input, net, useGPU, useMatConvnet )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    %input = single((sig_model/sig_f/255)*input);
    input = single((1/255)*input);
    %%% convert to GPU
    if useGPU && useMatConvnet
        input = gpuArray(input);
    end
    
    if useMatConvnet
        res = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    else
        res = simplenn_matlab(net, input); %%% use this if you did not install matconvnet.
    end
    
    output = input - res(end).x;
    
    %%% convert to CPU
    if useGPU && useMatConvnet
        output = gather(output);
    end
    
    output = double(255*output);
end

