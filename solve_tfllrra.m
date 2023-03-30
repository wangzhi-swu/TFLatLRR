function [Z,L,E,EE] = solve_tfllrra(X,A,B,lambda,rho,r1,r2,tol)

epsilon = 1e-10;
maxIter = 1e6;
[d n] = size(X);
nA = size(A,2);
dB = size(B,1);
max_mu = 1e6;
mu = 1e-6;
ata = A'*A;
bbt = B*B';
inv_ata = inv(ata+eye(nA));
inv_bbt = inv(bbt+eye(dB));
%% Initializing optimization variables
Z = zeros(nA,n);
L = zeros(d,dB);
E = sparse(d,n);

Y1 = zeros(d,n);
Y2 = zeros(nA,n);
Y3 = zeros(d,dB);

%% Start main loop
iter = 0;
display=1;
disp(['initial,r(Z)=' num2str(rank(Z)) ',r(L)=' num2str(rank(L)) ',|E|_1=' num2str(sum(sum(abs(E))))]);
while iter<maxIter
    iter = iter + 1;
     %update J
    temp = Z + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>epsilon));
    sigma(r1+1:svp,:) = sigma(r1+1:svp,:).*(mu/(2+mu));
    sigma = sigma(1:svp,:);
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    rZ = svp; %rank(J);
    
    %update S
    temp = L + Y3/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>epsilon));
    sigma(r2+1:svp,:) = sigma(r2+1:svp,:).*(mu/(2+mu));
    sigma = sigma(1:svp,:);
    S = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    rL = svp; %rank(S);
    
    %udpate Z
    hZ = A'*X-A'*L*B-A'*E+J+(A'*Y1-Y2)/mu;
    Z = inv_ata*hZ;
    
    %update L
    hL = X*B'-A*Z*B'-E*B'+S+(Y1*B'-Y3)/mu;
    L = hL*inv_bbt; 
    
    %update E
    temp = X-A*Z-L*B+Y1/mu;
     E = solve_l1l2(temp,lambda/mu);%lambda/mu
     
    %update the multiplies
    leq1 = X-A*Z-L*B-E;
    leq2 = Z-J;
    leq3 = L-S;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max(max(max(abs(leq3))),stopC);
    if display&&(iter==1 || mod(iter,50)==0 || stopC<tol)    
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ',stopALM=' num2str(stopC,'%2.3e') ...
            ',|E|_1=' num2str(sum(sum(abs(E))))]);
    end
    EE(iter)=stopC;
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq1;
        Y2 = Y2 + mu*leq2;
        Y3 = Y3 + mu*leq3;
        mu = min(max_mu,mu*rho);
    end
end
function [E] = solve_l1l2(W,lambda)
n = size(W,2);%W的列数
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end

function [x] = solve_l2(w,lambda)%lemma4.1
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);%返回2范式
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
