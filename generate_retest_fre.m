clear;close all;
clear variables;

addpath D:\caffe\caffe-windows-master\matlab
%% settings
folder = 'Test';
savepath = 'D:\caffe\caffe-windows-master\examples\denoise\wavelet_denoise\dehaar\test.h5';

size_input = 50;
size_label = 50;
% scale = 3;
stride = 10 ;
sigma = 25;
%% initialization
data = zeros(size_input/2, size_input/2, 4, 1);
label = zeros(size_label, size_label, 1, 1);
count = 0;
%% model
model_dir = 'D:\caffe\caffe-windows-master\examples\denoise\wavelet_denoise';
net_model = [model_dir '\denoise_mat.prototxt'];
net_weights = [model_dir '\shot\_iter_iter_500000.caffemodel'];
phase = 'test';
net = caffe.Net(net_model, net_weights, phase);
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
            subin=zeros(20,20,4);
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            [cA,cH,cV,cD]=dwt2(subim_input,'haar');
            subin(:,:,1)=cA;
            subin(:,:,2)=cH;
            subin(:,:,3)=cV;
            subin(:,:,4)=cD;
            subin={subin};
            im_label=net.forward(subin);
            im_label=cell2mat(im_label);       
            cA2=im_label(:,:,1);
            cH2=im_label(:,:,2);
            cV2=im_label(:,:,3);
            cD2=im_label(:,:,4);
            subim_label = im_gnd (x : x+size_label-1, y : y+size_label-1);
            count=count+1;
            data(:, :, 1, count) = cA;
            data(:, :, 2, count) = cH;
            data(:, :, 3, count) = cV;
            data(:, :, 4, count) = cD;
            label(:, :, 1, count) = subim_label;
        end
    end
end



order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, 1, order); 
%% writing to HDF5
chunksz = 2;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
