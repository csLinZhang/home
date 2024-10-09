function [tHist] = BGP(im)

% ========================================================================
% Copyright(c) 2012 Lin ZHANG, School of Software Engineering, Tongji University, 
% Shanghai, China 
% All Rights Reserved.
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereby
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% This is an implementation of the algorithm for calculating the
% Binary Gabor Pattern (BGP) histogram of the given image.
%----------------------------------------------------------------------
% Please refer to the following paper and the website for more details
%
% Lin Zhang,Zhiqiang Zhou, and Hongyu Li, "Binary Gabor pattern: an efficient 
% and robust descriptor for texture classification", in: Proc. IEEE International
% Conference on Imgae Processing, pp. 81-84, 2012, Orlando, USA.
%
% http://sse.tongji.edu.cn/linzhang/
%----------------------------------------------------------------------
%
% Input : im: the gray-scale image whose BGP histogram you want
% Output: tHist: the BGP histogram of the given image. It is a vector with 216 
% bins.

%the image need some preprocessing
im = double(im);
Ex = mean2(im);
sigma = std(im(:));
im = (im - Ex) ./ sigma;

[rows, cols] = size(im);
GA = gaborArray(rows, cols);
mapping = GetMaping(8); %8 bit; each bit for one orientation

halfLength = 17;
filterRows = 2*halfLength + 1;
filterCols = 2*halfLength + 1;
fftRows = rows + filterRows -1; %here we want to set the size of the FFT
fftCols = cols + filterCols -1;

noOrientatations = 8;
imageFFT = fft2(im, fftRows, fftCols);
tHist = zeros(216, 1); %8 orientations for each scale. That will generate 36 unique patterns for each scale
%we use 3 scales. So, the final histogram has 216 bins.

evenGaborRes = zeros(rows - halfLength * 2, cols - halfLength * 2, 8); 
oddGaborRes = zeros(rows - halfLength * 2, cols - halfLength * 2, 8); 
for scaleIndex = 1:3
    evenHistAtThisScale = zeros(36, 1);
    oddHistAtThisScale = zeros(36, 1);
    gabor = GA{scaleIndex};
    
    for oriIndex = 1:noOrientatations
        gaborResponse = ifft2(imageFFT .* gabor(:,:,oriIndex));
        %get the even and odd Gabor responses. And the response is the
        %'valid' part. 
        evenGaborRes(:,:,oriIndex) = real(gaborResponse(halfLength*2+1:rows, halfLength*2+1:cols));
        oddGaborRes(:,:,oriIndex) = imag(gaborResponse(halfLength*2+1:rows, halfLength*2+1:cols));
    end
    
    evenBinaryRes = evenGaborRes > 0;
    oddBinaryRes = oddGaborRes > 0;
    
    evenNoArray = zeros(rows - halfLength * 2, cols - halfLength * 2);
    oddNoArray = zeros(rows - halfLength * 2, cols - halfLength * 2);
    for oriIndex = 1:noOrientatations
        evenNoArray = evenNoArray + evenBinaryRes(:,:, oriIndex) * (2^(oriIndex-1));
        oddNoArray = oddNoArray + oddBinaryRes(:,:, oriIndex) * (2^(oriIndex-1));
    end
  
    [validRows, validCols] = size(evenNoArray);
    for row = 1:validRows
        for col = 1:validCols
            number = evenNoArray(row, col);
            pattern = mapping(number + 1);
            evenHistAtThisScale(pattern) = evenHistAtThisScale(pattern) + 1;
            
            number = oddNoArray(row, col);
            pattern = mapping(number + 1);
            oddHistAtThisScale(pattern) = oddHistAtThisScale(pattern) + 1;
        end
    end
    
    evenHistAtThisScale = evenHistAtThisScale / sum(evenHistAtThisScale);
    oddHistAtThisScale = oddHistAtThisScale / sum(oddHistAtThisScale);
    
    tHist((scaleIndex - 1) * 72 + 1: (scaleIndex - 1) * 72 + 36) = evenHistAtThisScale;
    tHist((scaleIndex - 1) * 72 + 37: scaleIndex * 72) = oddHistAtThisScale;
end
return;

%====================================
function gaborFFT = gaborArray(imageRows, imageCols)
% Bounding box
halfLength = 17;
%these two variables are used to extend the Gabor filter for the fft use.
fftRows = imageRows + (halfLength * 2 + 1) - 1;
fftCols = imageCols + (halfLength * 2 + 1) - 1;
noOrientatations = 8;
xmax = halfLength;
xmin = -halfLength; 
ymax = halfLength;
ymin = -halfLength;
[x,y] = meshgrid(xmin:xmax,ymin:ymax);

ratio = 1.82; %fixed
mask = ones(halfLength * 2 + 1, halfLength * 2 + 1);
for row = 1:halfLength * 2 + 1
    for col = 1:halfLength * 2 + 1
        if (row - halfLength)^2 + (col - halfLength)^2 > halfLength ^ 2
            mask(row,col) = 0;
        end
    end
end

sigma = 0.7; %fixed
lambda = 1.3; %fixed

gaborFirstScale = zeros(fftRows, fftCols, 8);
gaborSecondScale = zeros(fftRows, fftCols, 8);
gaborThirdScale = zeros(fftRows, fftCols, 8);
for oriIndex = 1:noOrientatations
    theta = pi / noOrientatations * (oriIndex - 1);
    % Rotation 
    x_theta=x*cos(theta)+y*sin(theta);
    y_theta=-x*sin(theta)+y*cos(theta);

    gb = exp(-.5*(x_theta.^2/sigma^2+y_theta.^2/(ratio * sigma)^2)).*cos(2*pi/lambda*x_theta);
    total = sum(sum(gb.*mask));
    meanInner = total / sum(sum(mask));
    gb = gb - mean2(meanInner);
    evenGabor = gb .* mask;
    
    gb = exp(-.5*(x_theta.^2/sigma^2+y_theta.^2/(ratio * sigma)^2)).*sin(2*pi/lambda*x_theta);
    total = sum(sum(gb.*mask));
    meanInner = total / sum(sum(mask));
    gb = gb - mean2(meanInner);
    oddGabor = gb .* mask;
    
    gaborFirstScale(:,:,oriIndex) = fft2(evenGabor + 1i * oddGabor, fftRows, fftCols);
end

sigma = 2.5;% fixed
lambda = 5.2;% fixed
for oriIndex = 1:noOrientatations
    theta = pi / noOrientatations * (oriIndex - 1);
    % Rotation 
    x_theta=x*cos(theta)+y*sin(theta);
    y_theta=-x*sin(theta)+y*cos(theta);

    gb = exp(-.5*(x_theta.^2/sigma^2+y_theta.^2/(ratio * sigma)^2)).*cos(2*pi/lambda*x_theta);
    total = sum(sum(gb.*mask));
    meanInner = total / sum(sum(mask));
    gb = gb - mean2(meanInner);
    evenGabor = gb .* mask;
    
    gb = exp(-.5*(x_theta.^2/sigma^2+y_theta.^2/(ratio * sigma)^2)).*sin(2*pi/lambda*x_theta);
    total = sum(sum(gb.*mask));
    meanInner = total / sum(sum(mask));
    gb = gb - mean2(meanInner);
    oddGabor = gb .* mask;
    gaborSecondScale(:,:,oriIndex) = fft2(evenGabor + 1i * oddGabor, fftRows, fftCols);
end

sigma = 4.5; %fixed
lambda = 22; %fixed
for oriIndex = 1:noOrientatations
    theta = pi / noOrientatations * (oriIndex - 1);
    % Rotation 
    x_theta=x*cos(theta)+y*sin(theta);
    y_theta=-x*sin(theta)+y*cos(theta);

    gb=exp(-.5*(x_theta.^2/sigma^2+y_theta.^2/(ratio * sigma)^2)).*cos(2*pi/lambda*x_theta);
    total = sum(sum(gb.*mask));
    meanInner = total / sum(sum(mask));
    gb = gb - mean2(meanInner);
    evenGabor = gb .* mask;
    
    gb=exp(-.5*(x_theta.^2/sigma^2+y_theta.^2/(ratio * sigma)^2)).*sin(2*pi/lambda*x_theta);
    total = sum(sum(gb.*mask));
    meanInner = total / sum(sum(mask));
    gb = gb - mean2(meanInner);
    oddGabor = gb .* mask;
    
    gaborThirdScale(:,:,oriIndex) = fft2(evenGabor + 1i * oddGabor, fftRows, fftCols);
end

gaborFFT = {};
gaborFFT{1} = gaborFirstScale;
gaborFFT{2} = gaborSecondScale;
gaborFFT{3} = gaborThirdScale;
return;
%==================================
function mapping = GetMaping(bitNo)
% if all the bits are zero, its pattern is named as "1". So all the
% patterns in this script are from 2.
%bitNo = 8;

bins = uint32(2 ^ bitNo);

mapping = zeros(bins,1);
labelMap = [];
tag = 2;
for index = 0:bins-1
    bitsForThisNumber = bitget(index, 1:bitNo);
    bitsForThisNumber = bitsForThisNumber';
   
    largestNumber = index;
    for shiftIndex = 1:bitNo - 1
        shiftedBits = circshift(bitsForThisNumber, shiftIndex);

        thisNum = 0;
        for i = 1:bitNo
           thisNum = thisNum + (2 ^ (i-1)) * shiftedBits(i);
        end
        if thisNum > largestNumber
           largestNumber = thisNum;
        end
    end
        
        if isempty(labelMap)
            labelMap = [labelMap; largestNumber 1];
            mapping(index + 1) = 1;
        else
            [row] = find(labelMap(:,1) == largestNumber);
            if isempty(row)
                labelMap = [labelMap; largestNumber tag];
                mapping(index + 1) = tag;
                tag = tag + 1;
            else
                mapping(index + 1) = labelMap(row, 2);
            end
        end
end
return;