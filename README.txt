This repository is made to publish code used in "Regularization by
Denoising: Clarifications and New Interpretations" by Reehorst, Schniter
https://arxiv.org/abs/1806.02296v2

In order to recreate figures 1-4 and tables 1-3 run the corresponding
script. figure_2.m and tables.m work with multiple denoisers. To change the
denoiser used, change the value of the "denoiser" variable within the
script file.

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
FASTA   https://github.com/tomgoldstein/fasta-matlab
BM3D    http://www.cs.tut.fi/~foi/GCF-BM3D/index.html#ref_software
TNRD    https://pan.baidu.com/s/1geNQP0J
DnCNN   https://github.com/cszn/DnCNN
nlm     https://www.mathworks.com/matlabcentral/fileexchange/52018-simple-non-local-means-nlm-filter
rwt     https://github.com/ricedsp/rwt
