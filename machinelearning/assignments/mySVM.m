%软间隔支持向量机，基于凸二次规划求解对偶问题的实现
%同济大学，张林，2022年10月

random = unifrnd(-1,1,50,2);

group1 = ones(50,2) + random; 
group1(50,:) = [4,5]; 
group2 = 3.5*ones(50,2) + random;
group2(50,:) = [1,0.1]; 

C = 3;

X=[group1;group2]; 
data=[group1,-1*ones(50,1);group2,1*ones(50,1)];

y=data(:,end);

Q=zeros(length(X),length(X));
for i=1:length(X)
    for j=1:length(X)
        Q(i,j)=X(i,:)*(X(j,:))'*y(i)*y(j);
    end
end
q=-1*ones(length(X),1);

A = [];
b= [];
Aeq = y'; 
beq = zeros(1,1); 
lb = zeros(length(X),1);
ub = ones(length(X),1)*C;

[alpha,fval]=quadprog(Q, q, A, b, Aeq,beq,lb,ub);


tooSmallIndex = alpha<1e-04;
alpha(tooSmallIndex) = 0;


w=0;
sumPartInb=0;
svIndices = find(alpha~=0); 
j = svIndices(1);
for i=1:length(svIndices) %
    w = w+alpha(svIndices(i))*y(svIndices(i))*X(svIndices(i),:)';
    sumPartInb = sumPartInb + alpha(svIndices(i))*y(svIndices(i))*(X(svIndices(i),:)*X(j,:)');
end
b = y(j)-sumPartInb;

figure
gscatter(X(:,1),X(:,2),y); 



supportVecs = X(svIndices,:);

hold on
plot(supportVecs(:,1),supportVecs(:,2),'ko','MarkerSize',10)

hold on
k=-w(1)./w(2); 
bb=-b./w(2);
xx=0:4;
yy=k.*xx+bb;
plot(xx,yy,'-')
hold on
yy=k.*xx+bb+1./w(2);
plot(xx,yy,'--')
hold on
yy=k.*xx+bb-1./w(2);
plot(xx,yy,'--')
title('support vector machine')
xlabel('dimension1')
ylabel('dimension2')
legend('group1','group2','support vector','separating hyperplane')