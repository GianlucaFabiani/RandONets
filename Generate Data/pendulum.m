function u_t=pendulum(t,u,k,forc)
u1=u(1,:); u2=u(2,:);
u1_t=u2;
u2_t=-k*sin(u1)+forc(t);
u_t=[u1_t;u2_t];
end
