clear;close all;
clear variables;

addpath D:\caffe\caffe-windows-master\matlab
%% settings
folder = 'Train';
savepath = 'train.h5';

size_input = 40;
size_label = 40;
% scale = 3;
stride = 40;
sigma = 25;
%% initialization
data = zeros(size_input/2, size_input/2, 4, 1);
label = zeros(size_label/2, size_label/2, 4, 1);
count = 0;
%% generate data
filepaths = dir(fullfile(folder,'*.jpg'));
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    im_gnd = im2double(image(:, :, 1));
    im_input = im_gnd + (sigma/255)*randn(size(im_gnd));
%     im_label = modcrop(image, scale);
    [hei,wid] = size(im_input);
    randn('seed', 0);                        
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            [cA,cH,cV,cD]=dwt2(subim_input,'haar');            
            subim_label = im_gnd (x : x+size_label-1, y : y+size_label-1);
            [cA2,cH2,cV2,cD2]=dwt2(subim_label,'haar');    
            count=count+1;
            data(:, :, 1, count) = cA;
            data(:, :, 2, count) = cH;
            data(:, :, 3, count) = cV;
            data(:, :, 4, count) = cD;
          
            label(:, :, 1, count) = cA2;
            label(:, :, 2, count) = cH2;
            label(:, :, 3, count) = cV2;
            label(:, :, 4, count) = cD2;
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 
%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);

