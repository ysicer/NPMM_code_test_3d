%BNU, Jun Liu, EMTV.
%modified from EMTV to NPMM by Shi Yan.
%function [Ima,phi,y] = NonParaEM_main(parr,inner,tau)
clc;clear;
close all;
addpath('auxcodes')

mex .\auxcodes\DensityFun.cpp
mex .\auxcodes\DensityFun_3d_2.cpp

filename = datestr(now,'mmmm_dd_yyyy__HH_MM_SS');
mkdir(filename)

Ima=imread('.\images\alldone_2.png');

parr = 1e-2;
inner = 2000;
tau = 5e-3;

[row,col,dim] = size(Ima);

Ima = double(Ima);
Ima = Ima - min(Ima(:));
Ima = Ima/(max(Ima(:)));
Ima = Ima;

%Ima = imresize(Ima,[64,64]);
[row,col,dim] = size(Ima);
Ima = double(Ima);

%Ima = imnoise(Ima,'Gaussian',0,1e-5);

%rng('default');
%noise = 50/255*randn([row,col]) + 0;
%Ima = Ima + noise;

%s = 1e-1;
%im_r = ricernd(Ima, s);
%Ima = im_r;

%Ima=round(mat2gray(Ima)*255);

imwrite(Ima,['.\' filename '\' 'noisy.tif'],'tiff','Resolution',300);
[M N H]=size(Ima);
MN=M*N;
K=2;

% used for initialize kmeans to a fixed class. (use the first victor of Ima)
Img = Ima(:,:,2);
for k=1:K
sp(k,1)=1/k*max(Img(:));
end
Lab=kmeans(Img(:),K,'start',sp);

for k=1:K
phi(:,:,k)=double(reshape(Lab,[M N])==k);
alpha(k)=1/K;
end

%u=rand(M,N)>0.5;
basic =repmat(0,[M N K]);
eta1=repmat(0,[M N K]);
eta2=eta1;

%tau = .01;
%parr = 70;
%inner = 100;

for i=1:10
    %phi=EStep(Ima,phi);
    %reg=1.2*Reg_term(phi,5);

%test
%eta1 = basic;
%eta2 = basic;
reg = parr*mydiv(eta1,eta2);

phiood = phi;
%phi(:,:,1) = double(phi(:,:,1)>phi(:,:,2));
%phi(:,:,2) = 1-phi(:,:,1);
%alpha(1) = .5;
%alpha(2) = .5;
%alpha(1) = sum(sum(phi(:,:,1)));
%alpha(2) = sum(sum(phi(:,:,2)));
figure(19),imshow(phi(:,:,1),[]),title(i)
    [phi,~,p]=NonPara3dEStep_smooth(Ima,phi,reg,alpha);
if i==1
%phi(:,:,1) = double(phi(:,:,1)>phi(:,:,2));
%phi(:,:,2) = 1-phi(:,:,1);
end


%if mod(i,1)==0
eta1=repmat(0,[M N K]);
eta2=eta1;
%end

I = Ima;

for j=1:inner %inner for \eta

[tempx,tempy] = mygradient(phi);
eta1 = eta1 - tau*tempx;
eta2 = eta2 - tau*tempy;
[eta1,eta2] = Proj(eta1,eta2,1);
reg = parr*mydiv(eta1,eta2);

pe=p.*exp(-reg);
for k=1:K
temp=pe(:,:,k)./(sum(pe,3)+eps);
phinew(:,:,k)=temp;
alpha(k)=sum(temp(:))/(M*N);
end
phi = phinew;

end

[t1 t2]=max(phi,[],3);
figure(15),
num1=ceil(K/2);
for k=1:K
subplot(num1,2,k),
imshow(t2==k);
hold on,
contour(phiood(:,:,1),[0.5,0.5],'r')
title(['Class ' num2str(k)]),drawnow;
imwrite((t2-1)/(K-1),['.\' filename '\' 'iter' num2str(i) 'par=' num2str(parr) 'dt=' num2str(tau) 'inner=' num2str(inner) '.png'])

end

end




x=0:255;
epsilon=2;

%{
for k=1:K
y(k,:)=DensityFun(Ima,x,phi(:,:,k),epsilon);
end

h=hist(Ima(:),256);
h=h/numel(Ima);
figure(17),
bar(h);
hold on;
plot(x,sum(y.*repmat(alpha',[1 size(y,2)]),1),'m');
hold off;
legend('Normalized Histogram',['p']);
%}

[t1 t2]=max(phi,[],3);
%figure(16),imshow(t2,[]);
%title('Final segmentation');

imwrite((t2-1)/(K-1),['.\' filename '\' 'Final' 'iter' num2str(i) 'par=' num2str(parr) 'dt=' num2str(tau) '.png'])

                     show3dspread(Ima,t2,2)
                     
                     
                     figure,imshow(Ima),hold on, contour(t2,[2,2],'r')
