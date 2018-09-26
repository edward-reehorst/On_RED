This repository is made to publish code used in "Regularization by
Denoising: Clarifications and New Interpretations" by Reehorst, Schniter
https://arxiv.org/abs/1806.02296v2

This code can be used to recreate tables 1-3 and figures 1-9.
The figures and tables can be reproduced by their corresponding script:
tables 1-3      tables.m (change value of denoiser for the different
                denoisers)
figure 1:       cost_vs_fp_error.m
figure 2:       RED_cost_visualization.m (change value of denoiser for the
                different denoisers)
figure 3:       ADMM_iteration_test.m
figure 4-6:     algorithm_comparison.m (denoiser set to 'TNRD')
figure 7-9:     algorithm_comparison.m (denoiser set to 'DWT')

This code uses several third-party code libraries. Directions on use and
links to the original code can be found below.

Using DnCNN:
DnCNN works much faster with matconvnet (http://www.vlfeat.org/matconvnet/)
Once matconvnet is installed add {matconvnet root folder}/matlab to the
matlab path and run vl_setupnn and make sure the useMatConvnet variable is
set to true. If matconvnet is compiled with GPU support, set useGPU = true,
else set to false. If you do not use matconvnet, set useMatConvnet = false
to use DnCNN

Using TNRD:
TNRD is only supported by windows and linux (not mac)

Using TDT (implemented using Rice-Wavelet-Toolbox):
Run rwt/bin/compile.m before using the rwt.

Links to original third-party code:
RED     https://github.com/google/RED
BM3D    http://www.cs.tut.fi/~foi/GCF-BM3D/index.html#ref_software
TNRD    https://pan.baidu.com/s/1geNQP0J
DnCNN   https://github.com/cszn/DnCNN
nlm     https://www.mathworks.com/matlabcentral/fileexchange/52018-simple-non-local-means-nlm-filter
rwt     https://github.com/ricedsp/rwt
