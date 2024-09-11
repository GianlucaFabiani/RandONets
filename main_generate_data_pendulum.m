clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
Nic=1000; %number of different functions (samples)
%
Nx=100; %number of points in the input grid (where I evaluate the function)
Ny0=100; %number of points in the output grid (where I evaluate the function)
Nv0=1;
Ny=Ny0*Nv0;
ddim=2;
%
x0=0; xf=1;
xx=linspace(x0,xf,Nx)';
yy0=linspace(x0,xf,Ny0)';
v00=-pi; v0f=pi;
if Nv0>1
    vv0=linspace(v00,v0f,Nv0);
    [VV0,YY0]=meshgrid(vv0,yy0);
    yy=[reshape(YY0,[],1),reshape(VV0,[],1)];
elseif Nv0==1
    vv0=0;
    ddim=1;
    yy=yy0;
end

DX=xf-x0;
DV=v0f-v00;
%
ff=zeros(Nx,Nic);
G=zeros(Ny,Nic);
tic;
for kic=1:Nic
    disp(kic/Nic)
    % oms=2*pi*(2*rand(1)-1);
    % As=(2*rand(1)-1);
    % ps=2*pi*rand(1);
    % omc=2*pi*(2*rand(1)-1);
    % Ac=(2*rand(1)-1);
    % pc=2*pi*rand(1);
    %
    ss=500*rand(1,200)/DX;
    cc=x0+DX.*rand(1,200);
    ww=2*rand(200,1)-1;
    a0=2*rand(1,1)-1;
    a1=2*rand(1,1)-1;
    a2=2*rand(1,1)-1;
    %f2=@(x) As*sin(oms*x+pc)+Ac*cos(omc*x+ps)+a0+a1*x+a2*x.^2+a3*x.^3+tanh(aa*(x-cc));
    %fx=@(x) As*oms*cos(oms*x+pc)-Ac*omc*sin(omc*x+ps)+a1+2*a2*x+3*a3*x.^2+aa*sech(aa*(x-cc)).^2;
    f=@(x) (exp(-ss.*(x-cc).^2)*ww+a0+a1*x+a2*x.^2)/20;
    ff(:,kic)=f(xx);
    kp=9.81;%9.81/0.1;
    pendulum_f=@(t,u) pendulum(t,u,kp,f);
    options=odeset('AbsTol',1e-12,'RelTol',1e-10);
    for kv0=1:Nv0
        v0=vv0(kv0);
        II0=(1:Ny0)+(kv0-1)*Ny0;
        [tttt,uint]=ode45(pendulum_f,yy0',[v0;0],options);
        G(II0,kic)=uint(:,1);
    end
end
toc
ntrain=round(Nic*80/100);
ntrain2=round(Nic*15/100);
II=randperm(Nic,ntrain);
II2=randperm(Nic,ntrain2);
%
G_train=G(:,II);
ff_train=ff(:,II);
G_train_few=G(:,II2);
ff_train_few=ff(:,II2);
%
G_test=G;
G_test(:,II)=[];
ff_test=ff;
ff_test(:,II)=[];
G_test_few=G;
G_test_few(:,II2)=[];
ff_test_few=ff;
ff_test_few(:,II2)=[];
%
%
figure(20)
subplot(1,2,1)
plot(xx,ff(:,1:min(20,Nic)))
subplot(1,2,2)
plot(yy(:,1),G(:,1:min(20,Nic)))
%
save('data_pendulum','G_train','G_test','ff_train','ff_test',...
    'G_train_few','G_test_few','ff_train_few','ff_test_few',...
    'xx','yy','ff','G','Nic','DX','ntrain','Nx','Ny','x0','xf',...
    'ddim','v00','v0f','DV','Ny0','Nv0')

%%%%%%%%%%%%%
%for python%
%#unaligned
%# size branch (samples,grid), size trunk (samples,sizeinp)
%# size output (samples,1)
%#just triple

ks=0;
np=length(II);
Gp_train=zeros(Ny*np,1);
ffp_train=zeros(Ny*np,Nx);
yyp_train=zeros(Ny*np,ddim);

for s=1:np
    for k=1:Ny
        ks=ks+1;
        Gp_train(ks,1)=G_train(k,s);
        ffp_train(ks,:)=ff_train(:,s)';
        yyp_train(ks,:)=yy(k,:);
    end
end

ks=0;
np_inv=Nic-length(II);
Gp_test=zeros(Ny*np_inv,1);
ffp_test=zeros(Ny*np_inv,Nx);
yyp_test=zeros(Ny*np_inv,ddim);

for s=1:np_inv
    for k=1:Ny
        ks=ks+1;
        Gp_test(ks,1)=G_test(k,s);
        ffp_test(ks,:)=ff_test(:,s)';
        yyp_test(ks,:)=yy(k,:);
    end
end

ks=0;
np2=length(II2);
Gp_train_few=zeros(Ny*np2,1);
ffp_train_few=zeros(Ny*np2,Nx);
yyp_train_few=zeros(Ny*np2,ddim);

for s=1:np2
    for k=1:Ny
        ks=ks+1;
        Gp_train_few(ks,1)=G_train_few(k,s);
        ffp_train_few(ks,:)=ff_train_few(:,s)';
        yyp_train_few(ks,:)=yy(k,:);
    end
end

ks=0;
np2_inv=Nic-length(II2);
Gp_test_few=zeros(Ny*np2_inv,1);
ffp_test_few=zeros(Ny*np2_inv,Nx);
yyp_test_few=zeros(Ny*np2_inv,ddim);

for s=1:np2_inv
    for k=1:Ny
        ks=ks+1;
        Gp_test_few(ks,1)=G_test_few(k,s);
        ffp_test_few(ks,:)=ff_test_few(:,s)';
        yyp_test_few(ks,:)=yy(k,:);
    end
end

save('data_pendulum_forpython','Gp_train','Gp_test','ffp_train','ffp_test',...
    'Gp_train_few','Gp_test_few','ffp_train_few','ffp_test_few',...
    'xx','yyp_train','yyp_test','yyp_train_few','yyp_test_few','yy','ff','G',...
    'Nic','DX','ntrain','ntrain2','Nx','Ny','x0','xf','np','np2',...
    'ddim','v00','v0f','DV','Ny0','Nv0')