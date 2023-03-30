close all;
clear all;
clc;
addpath('.\measure');
addpath('.\Database');
% load('EYB_Group1.mat');
% load('EYB_Group2.mat');
% load('EYB_Group3.mat');
load('EYB_Group4.mat');
X=mapminmax(X,0,1);%0-1
gnd=labels;
K=max(gnd);
[d n]=size(X);


%% Parameters
rho=1.5;
alpha=3;
r1=40;
r2=40;
lambda=1e-6;
tol=1e-8;
%% segmentation 

    [Z,L,E,EE] = solve_tfllrr(X,lambda,rho,r1,r2,tol);
    
%% postprocessing
    [U s V] = svd(Z);
    s = diag(s);
    r = sum(s>1e-6);

    U = U(:, 1 : r);
    s = diag(s(1 : r));
    V = V(:, 1 : r);

    M = U * s.^(1/2);
    mm = normr(M);
    rs = mm * mm';
    L = rs.^(2 * alpha);
    
%% spectral clustering NCut
    idx = spectral_clustering(L, K);
    Y=gnd;
    predY=idx;
    [result,bestY] = Clustering8Measure(Y, predY);
    disp(['ACC nmi Purity Fscore Precision Recall AR Entropy=' num2str(result)]);
