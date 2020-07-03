load('X_iris');
X1 = X_iris(:,1:50);
Xtr1 = X1(:,1:35);
Xte1 = X1(:,36:50);
X2 = X_iris(:,51:100);
Xtr2 = X2(:,1:35);
Xte2 = X2(:,36:50);
X3 = X_iris(:,101:150);
Xtr3 = X3(:,1:35);
Xte3 = X3(:,36:50);
K=5;

Ptr=Xtr1;
Ntr=[Xtr2 Xtr3];
Xtr=[Ptr Ntr];
y=[ones(35,1);-ones(70,1)];
x0 = zeros(5,1);
% [ws1,C2_1] = LRBC_GD('f_LRBC','g_LRBC',x0,1e-6, Xtr, y);
 [ws1,C2_1] = LRBC_newton(Xtr,y,K);
disp(C2_1);


Ptr=Xtr2;
Ntr=[Xtr1 Xtr3];
Xtr=[Ptr Ntr];
y=[ones(35,1);-ones(70,1)];
[ws2,C2_2] = LRBC_newton(Xtr,y,K);


Ptr=Xtr3;
Ntr=[Xtr1 Xtr2];
Xtr=[Ptr Ntr];
y=[ones(35,1);-ones(70,1)];
[ws3,C2_3] = LRBC_newton(Xtr,y,K);
disp("Ptr");
disp(Ptr);
disp("Ntr");
disp(Ntr);

disp("ws without normalizing");
disp(ws1);
disp("ws without normalizing");
disp(ws2);
disp("ws without normalizing");
disp(ws3);

w1=ws1(1:4);
disp(size(w1));
b1=ws1(5);
w2=ws2(1:4);
b2=ws2(5);
w3=ws3(1:4);
b3=ws3(5);

disp(norm(w1));
disp(norm(w2));
disp(norm(w3));

% w1=ws1(1:4)/norm(ws1(1:4));
% b1=ws1(5)/norm(ws1(1:4));
% w2=ws2(1:4)/norm(ws2(1:4));
% b2=ws2(5)/norm(ws2(1:4));
% w3=ws3(1:4)/norm(ws3(1:4));
% b3=ws3(5)/norm(ws3(1:4));
% 
Ws=[w1 w2 w3]
disp(norm(Ws));
bs=[b1 b2 b3]';

% E = zeros(3,105);
E = zeros(3,45);
disp(E);
% 
% Xte=[Xte1 Xte2 Xte3];
% for i=1:length(Xte)
%     xi=Xte(:,i);
%     ti=Ws'*xi+bs
%     [~,ind]=max(ti);
%     E(ind,i) = 1;
% end

X_tr=[Xtr1 Xtr2 Xtr3];
for i=1:length(X_tr)
    xi=X_tr(:,i);
    ti=Ws'*xi+bs;
    [~,ind]=max(ti);
    E(ind,i) = 1;
end
% 
% E1= E(:,1:15);c1 = sum(E1')';
% E2= E(:,16:30);c2 = sum(E2')';
% E3= E(:,31:45);c3 = sum(E3')';
E1= E(:,1:35);c1 = sum(E1')';
E2= E(:,36:70);c2 = sum(E2')';
E3= E(:,71:105);c3 = sum(E3')';

disp('Confusion matrix');
C=[c1 c2 c3]

num_correct = trace(C)
% acc = num_correct/length(Xte)
acc = num_correct/length(X_tr)
acc = num_correct/sum(C,'all');
