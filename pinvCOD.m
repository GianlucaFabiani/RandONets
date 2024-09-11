function [Qr,T11,Vp]=pinvCOD(PHI,tol,matrixat)
if nargin<3
    matrixat='leftt'; %Ax=b
end
if nargin==1
    tol=max(eps(norm(PHI)*max(size(PHI)))*1e-3,1e-13);
end
if isempty(tol)
    tol=max(eps(norm(PHI)*max(size(PHI)))*1e-3,1e-13);
end
[m,n]=size(PHI);
flag=0;
if sum((matrixat-'right').^2)==0
    PHI=PHI';
    m1=m;
    m=n;
    n=m1;
    flag=1;
end
[Q,R,P]=qr(PHI,0);
Pm=zeros(n);
for i=1:n
    Pm(P(i),i)=1;
end
Rr=diag(R);
r=find(abs(Rr)>tol);
r=r(end);
Qr=Q(:,1:r);
T1=R(1:r,1:r);
T2=R(1:r,r+1:end);
Tadj=[T1';T2'];
[V,T11adj]=qr(Tadj,0);
T11=T11adj';
Vp=Pm*V;
if flag==1
    T11=T11';
    tempQr=Qr;
    Qr=Vp;
    Vp=tempQr;
end

end