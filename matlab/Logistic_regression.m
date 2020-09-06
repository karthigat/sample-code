x1=linspace(-1.5,2.5,200); %intial points
w=[-1.3764;1.5014;1.0892;1.3155;0.0711;-0.1547];
A=w(5);
B=w(2)+(w(4)*x1);
C=w(1)*x1+w(3)*x1.^2+w(6);
x2=(-B+sqrt(B.^2-(4*A*C)))/(2*A);
disp(x2);
X1=x1(:,42:200);
X2=x2(:,42:200);
real(x2);
p=[X1;X2]   
P=p;
x=[[-1 0 2 0 1  2];[-1 1 0 -2 0 -1]];
P_X = f(:,1:3)
N_X = f(:,4:6)
X = [P_X N_X];
xt1=[-0.5;-1.5];
xt2=[1.5;0];
X_1=[xt1 xt2];
plot(P(1,:),P(2,:),'k-','linew',2)
hold on
scatter(P_X(1,:),P_X(2,:),'o','b','filled');
scatter(N_X(1,:),N_X(2,:),'o','g','filled');
scatter(-0.5,-1.5,'o','k','filled'); %
scatter(1.5,0,'o','r','filled');
legend('decision boundary','P(1)','N(-1)','Xt1(N)','Xt2(P)')
hold off



