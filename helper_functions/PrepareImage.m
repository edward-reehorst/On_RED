% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

% returns the luminance channel of the input_image

function luma_image = PrepareImage(input_image)

if  size(input_image,3) == 3
    ycbcr_im = 255*rgb2ycbcr(input_image/255);
    luma_image = ycbcr_im(:,:,1);    
else
    luma_image = input_image;    
end

luma_image = double(luma_image);

