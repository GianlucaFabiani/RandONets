clc
clear
close all
set(0,'DefaultLineLineWidth',2)
%
load('data_antiderivative.mat')
%load('data_Pendulum.mat')
%load('data_DiffReac.mat')
%load('data_burgers.mat')
%load('data_AllenCahn.mat')
%
rng(5)
%
Nt=200; %number of neurons in the trunk hidden layer
NNb=[10,20,40,80,100,150,300,500,1000,2000]; %number of neurons in the branch hidden layer
%data_case='fews'; iters=10;
data_case='many'; iters=3; %Iterations are used solely to improve computational time estimation.
%

if sum((data_case-'fews').^2)==0
    G_test=G_test_few;
    G_train=G_train_few;
    ff_train=ff_train_few;
    ff_test=ff_test_few;
end

%
knb=0;
for Nb=NNb
    knb=knb+1;
    knb/length(NNb)
    for kmodel=1:2
        pause(0.0001)
        if kmodel==1
            tr_fB=@(x) x; flag_branch=1; %%%%%%%JL
        elseif kmodel==2
            tr_fB=@(x) cos(x); flag_branch=0;  %%%%%%RFFN 
        end
        timetime=0;
        for it=1:iters
        tstart=tic;
        RandONet=train_RandONet(ff_train,yy,G_train,Nt,Nb,kmodel);
        timetime=timetime+toc(tstart);
        end
        timetime=timetime/iters;
        %train set
        Gnet_train=eval_RandONet(RandONet,ff_train,yy);
        %test set
        Gnet_test=eval_RandONet(RandONet,ff_test,yy);
        if kmodel==1
            errMSE_JL_test(knb)=mean(mean((G_test-Gnet_test).^2));
            errmax_JL_test(knb)=max(max(abs(G_test-Gnet_test)));
            errL2_JL_test(knb,:)=sqrt(sum((G_test-Gnet_test).^2));
            time_JL(knb)=timetime;
        elseif kmodel==2
            errMSE_RFFN_test(knb)=mean(mean((G_test-Gnet_test).^2));
            errmax_RFFN_test(knb)=max(max(abs(G_test-Gnet_test)));
            errL2_RFFN_test(knb,:)=sqrt(sum((G_test-Gnet_test).^2));
            time_RFFN(knb)=timetime;
        end
    end
%
end
errmL2_RFFN_test=median(errL2_RFFN_test,2)';
err95L2_RFFN_test=prctile(errL2_RFFN_test,95,2)';
err05L2_RFFN_test=prctile(errL2_RFFN_test,5,2)';
%
errmL2_JL_test=median(errL2_JL_test,2)';
err95L2_JL_test=prctile(errL2_JL_test,95,2)';
err05L2_JL_test=prctile(errL2_JL_test,5,2)';

figure(1)
hold off
fill([NNb,fliplr(NNb)],[err05L2_JL_test,fliplr(err95L2_JL_test)],...
    'c','FaceAlpha',0.3,'LineStyle','none')
set(gca,'Xscale','log','Yscale','log')
hold on
loglog(NNb,errmL2_JL_test,'o-b')
loglog(NNb,errMSE_JL_test,'o--b')
hold on
fill([NNb,fliplr(NNb)],[err05L2_RFFN_test,fliplr(err95L2_RFFN_test)],...
    'm','FaceAlpha',0.3,'LineStyle','none')

loglog(NNb,errmL2_RFFN_test,'x-r','MarkerSize',8)

loglog(NNb,errMSE_RFFN_test,'x--r','MarkerSize',8)
grid on
set(gca,'FontSize',18,'YTick',10.^(-32:4:32))
xlabel('$M$ (branch neurons)','Interpreter','latex')
ylabel('error')
legend('JL 5\%-95\% $L^2$','JL median $L^2$','JL MSE',...
    'RFFN 5\%-95\% $L^2$',...
    'RFFN median $L^2$',...
    'RFFN MSE',...
    'interpreter','latex',...
    'NumColumns',2,'FontSize',14)
% legend('JL max-$L^2$','JL MSE','JL MaxAE',...
%     'RFFN max-$L^2$','RFFN MSE','RFFN MaxAE','interpreter','latex',...
%     'NumColumns',2,'FontSize',14)
VV=axis;
ylim(10.^[log10(VV(3))*1.1,log10(VV(4))+(log10(VV(4))-log10(VV(3)))/4])

figure(2)
hold off
fill([time_JL,fliplr(time_JL)],[err05L2_JL_test,fliplr(err95L2_JL_test)],...
    'c','FaceAlpha',0.3,'LineStyle','none');
set(gca,'Xscale','log','Yscale','log')
hold on
loglog(time_JL,errmL2_JL_test,'o-b')
loglog(time_JL,errMSE_JL_test,'o--b')
hold on
fill([time_RFFN,fliplr(time_RFFN)],[err05L2_RFFN_test,fliplr(err95L2_RFFN_test)],...
    'm','FaceAlpha',0.3,'LineStyle','none')

loglog(time_RFFN,errmL2_RFFN_test,'x-r','MarkerSize',8)

loglog(time_RFFN,errMSE_RFFN_test,'x--r','MarkerSize',8)
grid on
set(gca,'FontSize',18,'YTick',10.^(-32:4:32))
xlabel('time [seconds]','Interpreter','latex')
ylabel('error')
legend('JL 5\%-95\% $L^2$','JL median $L^2$','JL MSE',...
    'RFFN 5\%-95\% $L^2$',...
    'RFFN median $L^2$',...
    'RFFN MSE',...
    'interpreter','latex',...
    'NumColumns',2,'FontSize',14)
%VV=axis;
ylim(10.^[log10(VV(3))*1.1,log10(VV(4))+(log10(VV(4))-log10(VV(3)))/4])