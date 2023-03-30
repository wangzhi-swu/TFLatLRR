function [Z,L,E,EE] = solve_tfllrr(X,lambda,rho,r1,r2,tol)
Q1 = orth(X');
Q2 = orth(X);
A = X*Q1;
B = Q2'*X;

[Z,L,E,EE] = solve_tfllrra(X,A,B,lambda,rho,r1,r2,tol);

Z = Q1*Z;
L = L*Q2';