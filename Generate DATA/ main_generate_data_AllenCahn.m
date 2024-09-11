clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
rng(5)
%
Nic=3000; %number of different functions (samples)
%
Nx=100; %number of points in the input grid (where I evaluate the function)
Ny=100; %number of points in the output grid (where I evaluate the function)
%
x0=-1; xf=1;
xx=linspace(x0,xf,Nx)';
yy=linspace(x0,xf,Ny)';
DX=xf-x0;
%
ff=zeros(Nx,Nic);
G=zeros(Nx,Nic);
Reac=0;
nu=0.01;
Trasp=0;
NLAdv=0;
Phase=1;
for kic=1:Nic
    %clc
    kic/Nic
    oms=0;%2*pi*(2*rand(1)-1);
    As=0;%(2*rand(1)-1);
    ps=0;%2*pi*rand(1);
    % omc=2*pi*(2*rand(1)-1);
    % Ac=(2*rand(1)-1);
    % pc=2*pi*rand(1);
    %
    ss=50*rand(1,200)/DX;
    mu=x0+DX.*rand(1,200);
    ww=(2*rand(200,1)-1)/20;
    a0=(2*rand(1,1)-1)/20;
    a1=(2*rand(1,1)-1)/20;
    a2=(2*rand(1,1)-1)/20; 
    %f2=@(x) As*sin(oms*x+pc)+Ac*cos(omc*x+ps)+a0+a1*x+a2*x.^2+a3*x.^3+tanh(aa*(x-cc));
    %fx=@(x) As*oms*cos(oms*x+pc)-Ac*omc*sin(omc*x+ps)+a1+2*a2*x+3*a3*x.^2+aa*sech(aa*(x-cc)).^2;
    f=@(x) exp(-ss.*(x-mu).^2)*ww+a0+a1*x+a2*x.^2+As*sin(oms*x+ps);
    fx=@(x) (-2*ss.*(x-mu).*exp(-ss.*(x-mu).^2))*ww+a1+2*a2*x+As*oms*cos(oms*x+ps);
    fxx=@(x) ( exp(-ss.*(x-mu).^2).*((-2*ss.*(x-mu)).^2-2*ss)  )*ww+2*a2-As*oms^2*sin(oms*x+ps);
    ff(:,kic)=f(xx);
    G(:,kic)=nu*fxx(yy)+Trasp*fx(yy)+Reac*f(yy)+Phase*(f(yy)-f(yy).^3)+NLAdv*f(yy).*fx(yy);
end
figure
plot(yy,G)
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
plot(xx,ff(:,1:min(20,Nic)))
%
save('data_AllenCahn','G_train','G_test','ff_train','ff_test',...
    'G_train_few','G_test_few','ff_train_few','ff_test_few',...
    'xx','yy','ff','G','Nic','DX','ntrain','Nx','Ny','x0','xf')

%%%%%%%%%%%%%
%for python%
%#unaligned
%# size branch (samples,grid), size trunk (samples,sizeinp)
%# size output (samples,1)
%#just triple

ks=0;
np=length(II);
Gp_train=zeros(Nx*np,1);
ffp_train=zeros(Nx*np,Nx);
yyp_train=zeros(Nx*np,1);

for s=1:np
    for k=1:Nx
        ks=ks+1;
        Gp_train(ks,1)=G_train(k,s);
        ffp_train(ks,:)=ff_train(:,s)';
        yyp_train(ks,1)=yy(k);
    end
end

ks=0;
np_inv=Nic-length(II);
Gp_test=zeros(Nx*np_inv,1);
ffp_test=zeros(Nx*np_inv,Nx);
yyp_test=zeros(Nx*np_inv,1);

for s=1:np_inv
    for k=1:Nx
        ks=ks+1;
        Gp_test(ks,1)=G_test(k,s);
        ffp_test(ks,:)=ff_test(:,s)';
        yyp_test(ks,1)=yy(k);
    end
end

ks=0;
np2=length(II2);
Gp_train_few=zeros(Nx*np2,1);
ffp_train_few=zeros(Nx*np2,Nx);
yyp_train_few=zeros(Nx*np2,1);

for s=1:np2
    for k=1:Nx
        ks=ks+1;
        Gp_train_few(ks,1)=G_train_few(k,s);
        ffp_train_few(ks,:)=ff_train_few(:,s)';
        yyp_train_few(ks,1)=yy(k);
    end
end

ks=0;
np2_inv=Nic-length(II2);
Gp_test_few=zeros(Nx*np2_inv,1);
ffp_test_few=zeros(Nx*np2_inv,Nx);
yyp_test_few=zeros(Nx*np2_inv,1);

for s=1:np2_inv
    for k=1:Nx
        ks=ks+1;
        Gp_test_few(ks,1)=G_test_few(k,s);
        ffp_test_few(ks,:)=ff_test_few(:,s)';
        yyp_test_few(ks,1)=yy(k);
    end
end


save('data_AllenCahn_forpython','Gp_train','Gp_test','ffp_train','ffp_test',...
    'Gp_train_few','Gp_test_few','ffp_train_few','ffp_test_few',...
    'xx','yyp_train','yyp_test','yyp_train_few','yyp_test_few',...
    'yy','ff','G','Nic','DX','ntrain','ntrain2','Nx','Ny','x0','xf','np','np2')
