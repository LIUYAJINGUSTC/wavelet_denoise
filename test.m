 close all;
clear variables;
%% read ground truth image
% im  = imread('Set5\butterfly_GT.bmp');
addpath D:\caffe\caffe-windows-master\matlab
folder = 'test3';
filepaths = dir(fullfile(folder,'*.png'));
folder2 = './shot50';
filepaths2 = dir(fullfile(folder2, '*.caffemodel'));
N = length(filepaths2);
%iter = zeros(N, 1);			% 循环次数
psnr_line = zeros(1,N);		% 峰值信噪比
ii_count = 0;
for j=1:N
    caffe.reset_all();
for i = 1 : length(filepaths)
name = fullfile(folder,filepaths(i).name);
im =imread(fullfile(folder,filepaths(i).name));
if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end
[hei,wid]=size(im);
filename = 'denoise_mat.prototxt';
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
model_dir = 'D:\caffe\caffe-windows-master\examples\denoise\wavelet2_denoise';
net_model = [model_dir '\denoise_matran.prototxt'];
net_weights = [model_dir '\shot50\' filepaths2(j).name];
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

subin={subin};
im_label=net.forward(subin);
%		   im_label=net.forward(subimA_label,subimH_label,subimV_label,subimD_label);
im_label=cell2mat(im_label);
         
cA2=im_label(:,:,1);
cH2=im_label(:,:,2);
cV2=im_label(:,:,3);
cD2=im_label(:,:,4);

im_label=idwt2(cA2,cH2,cV2,cD2,'haar');
%% compute PSNR

label = uint8(im_label * 255);

im_gnd = uint8(im_gnd * 255);
psnr(1,i) = compute_psnr(im_gnd,label);


%% show results
fprintf('PSNR for %s CNN Reconstruction: %f dB\n', name,psnr(1,i));
% figure, imshow(im_input); title('noisy');
% 
% figure, imshow(im_gnd); title('noisy');
% figure, imshow(label); 
% title('CNN Reconstruction');
% 
% imwrite(label, ['CNN Reconstruction' '.bmp']);
end;
ii_count = ii_count +1;
psnr_line(1,j)=mean(psnr);
x(1,j)=ii_count*10000;
end;
plot(x,psnr_line);