%% Brain Matter Segmentation Pipeline
[filename, pathname] = uigetfile('*.dcm', 'Select DICOM CT Scan');
if isequal(filename,0), error('No file selected'); end

% 1. HU Conversion
ctInfo = dicominfo(fullfile(pathname, filename));
ctVolume = dicomread(ctInfo);
ctHU = double(ctVolume) * ctInfo.RescaleSlope + ctInfo.RescaleIntercept;
midSlice = ctHU(:,:,round(size(ctHU,3)/2));

% 2. Bone Removal
thresh = multithresh(midSlice(midSlice>50), 2);
boneMask = midSlice > thresh(2);
closedMask = imclose(boneMask, strel('disk',4));
finalBoneMask = imdilate(closedMask, strel('disk',7));
brainOnly = midSlice;
brainOnly(finalBoneMask) = NaN;

% 3. CLAHE + Bilateral Filter
brainForCLAHE = uint8(255 * mat2gray(brainOnly, [0 80])); % Narrower window for GM
claheEnhanced = adapthisteq(brainForCLAHE, 'ClipLimit',0.01, 'NumTiles',[10 10]);
filtered = imbilatfilt(double(claheEnhanced), 'DegreeOfSmoothing',15);

% 4. Gray Matter Segmentation
% Optimal HU range for gray matter (30-50 HU)
gmMask = filtered > 30 & filtered < 50; 
gmMask = bwareaopen(gmMask, 100); % Remove small regions
gmMask = imfill(gmMask, 'holes');

% 5. Create Final Segmentation
finalSeg = zeros(size(midSlice), 'like', midSlice);
finalSeg(gmMask) = midSlice(gmMask); % Preserve original HU values
finalSeg(finalBoneMask) = 0; % Keep bone areas black

% Visualization
figure('Position', [100 100 1400 400]);
subplot(1,4,1); imshow(midSlice, [0 80]); title('Original CT');
subplot(1,4,2); imshow(filtered, []); title('Enhanced Image');
subplot(1,4,3); imshow(gmMask); title('Gray Matter Mask');
subplot(1,4,4); imshow(finalSeg, [30 50]); 
title({'Segmented Gray Matter', '[30-50 HU Range]'});
colormap(gray); colorbar;