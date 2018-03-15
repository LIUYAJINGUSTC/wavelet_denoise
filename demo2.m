% =========================================================================
% Test code for Super-Resolution Convolutional Neural Networks (SRCNN)
%
% Reference
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, 
%   in Proceedings of European Conference on Computer Vision (ECCV), 2014
%
%   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks,
%   arXiv:1501.00092
%
% Chao Dong
% IE Department, The Chinese University of Hong Kong
% For any question, send email to ndc.forward@gmail.com
% =========================================================================

%% 设置工作路径
% path = which('demo_SR'); i = strfind(path, '\');
% path = path(1 : i(end)); cd(path);

%%
close all;
clear variables;
%% read ground truth image
% im  = imread('Set5\butterfly_GT.bmp');
addpath D:\caffe\caffe-windows-master\matlab
folder = 'test3';
filepaths = dir(fullfile(folder,'*.png'));
for i = 1 : length(filepaths)
name = fullfile(folder,filepaths(i).name);
im =imread(fullfile(folder,filepaths(i).name));
% im  = imread('zebra.bmp');
% im  = im(1:390,1:586);
% im = im(150:250,150:250);
% size_input = 22;
% size_label = 22;
% % scale = 3;
% stride = 22 ;
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end
[hei,wid]=size(im);
filename = 'denoise_mat_future.prototxt';
fid=fopen(filename,'r');
newline=fread(fid,'*char');
if size(num2str(wid/2),2)==3 newline(67:69)=num2str(wid/2);
elseif size(num2str(wid/2),2)==2 newline(67:68)=num2str(wid/2);
end
if size(num2str(hei/2),2)==3 newline(86:88)=num2str(hei/2);
elseif size(num2str(hei/2),2)==2 newline(86:87)=num2str(hei/2);
end  
fclose(fid);
f=fopen('denoise_matran.prototxt','w');
fwrite(f,newline);
fclose(f);
model_dir = 'D:\caffe\caffe-windows-master\examples\denoise\wavelet_denoise';
net_model = [model_dir '\denoise_matran.prototxt'];
net_weights = [model_dir '\_iterfre_iter_50000.caffemodel'];
phase = 'test';
net = caffe.Net(net_model, net_weights, phase);

sigma = 25;
%% work on illuminance only



im_gnd = single(im)/255;
subin=zeros(hei/2,wid/2,4);
randn('seed', 0);                          %% generate seed
im_input   = im_gnd + (sigma/255)*randn(size(im_gnd)); %% create a noisy image
[cA,cH,cV,cD]=dwt2(im_input,'haar');
subin(:,:,1)=cA;
subin(:,:,2)=cH;
subin(:,:,3)=cV;
subin(:,:,4)=cD;
figure
imshow(uint8(wcodemat(cA,255)));
figure
imshow(uint8(wcodemat(cH,255)));
figure
imshow(uint8(wcodemat(cV,255)));
figure
imshow(uint8(wcodemat(cD,255)));
subin={subin};
im_label=net.forward(subin);
%		   im_label=net.forward(subimA_label,subimH_label,subimV_label,subimD_label);
im_label=cell2mat(im_label);
%             subimH_label=zeros(size(subimH_label));
%             subimV_label=zeros(size(subimV_label));
%             subimD_label=zeros(size(subimD_label));
%             im_label=idwt2(subimA_label,subimH_label,subimV_label,subimD_label,'haar');
%             sub_label=idwt2(cA,subimH_label,cV,subimD_label,'haar');           
cA2=im_label(:,:,1);
cH2=im_label(:,:,2);
cV2=im_label(:,:,3);
cD2=im_label(:,:,4);
figure
imshow(uint8(wcodemat(cA2,255)));
figure
imshow(uint8(wcodemat(cH2,255)));
figure
imshow(uint8(wcodemat(cV2,255)));
figure
imshow(uint8(wcodemat(cD2,255)));
im_label=idwt2(cA2,cH2,cV2,cD2,'haar');
%% compute PSNR

% label = modcrop(label, 13);
% im_input = modcrop(im_input, 13);
% 
% up_scale = 14;
label = uint8(im_label * 255);
% label = uint8(label * 255);
% im_gnd = shave(uint8(im_gnd * 255), [6, 6]);
% im_gnd = uint8(im_gnd(1:374,1:572) * 255);
im_gnd = uint8(im_gnd * 255);
psnr_srcnn(1,i) = compute_psnr(im_gnd,label);
% psnr_srcnn = compute_psnr(uint8(im_gnd(7e:nd, 7:end) *255) ,uint8(label(7:end, 7:end)) * 255);
figure, imshow(im_gnd-label);
%% show results
fprintf('PSNR for %s CNN Reconstruction: %f dB\n', name,psnr_srcnn(1,i));
figure, imshow(im_input); title('noisy');
% figure, imshow(im_input(7:end, 7:end)); title('noisy');
figure, imshow(im_gnd); title('noisy');
figure, imshow(label); 
title('CNN Reconstruction');

imwrite(label, ['CNN Reconstruction' '.bmp']);
end;
a=mean(psnr_srcnn);