#include <iostream>
#include <cassert>
#include <cmath>
#include "../MatrixUtils.h"
#include "../GaussSolver.h"
#include "../LUDecomposition.h"
#include "../QRDecomposition.h"
#include "../IterativeSolver.h"
#include "../EigenSolver.h"
#include "../MiscMath.h"

bool approx(double a, double b, double eps = 1e-5) {
    return std::abs(a - b) < eps;
}

int main() {
    std::cout << "Running tests...\n";
    
    // Gauss
    Matrix A(2,2); A.a = {{2,1},{1,3}};
    Vector b = {5,6};
    Vector x = solveGauss(A,b,true);
    assert(approx(x[0], 1.8) && approx(x[1], 1.4));  // <-- ИСПРАВЛЕНО
    std::cout << "Gauss OK\n";
    
    // LU
    Matrix A2(3,3); A2.a = {{4,3,2},{6,3,1},{2,1,3}};
    Vector b2 = {1,2,3};
    Matrix L,U; std::vector<int> P; int sw;
    decomposeLU(A2,L,U,P,sw);
    Vector x2 = solveLU(L,U,P,b2);
    assert(vectorNormMax(vecSub(mul(A2,x2),b2)) < 1e-5);
    std::cout << "LU OK\n";
    
    // QR
    Matrix A3(2,2); A3.a = {{1,1},{0,1}};
    Matrix Q,R; decomposeQR(A3,Q,R);
    Matrix QtQ = mul(transpose(Q),Q);
    assert(approx(QtQ.a[0][0],1.0) && approx(QtQ.a[1][1],1.0));
    assert(approx(QtQ.a[0][1],0.0) && approx(QtQ.a[1][0],0.0));
    std::cout << "QR OK\n";
    
    // Iterative
    Matrix A4(3,3); A4.a = {{5,2,1},{2,6,1},{1,1,4}};
    Vector b4 = {7,9,5};
    Vector xJ = solveJacobi(A4,b4,Vector(3,0.0),1e-6,1000);
    Vector xGS = solveGaussSeidel(A4,b4,Vector(3,0.0),1e-6,1000);
    assert(vectorNormMax(vecSub(mul(A4,xJ),b4)) < 1e-3);
    assert(vectorNormMax(vecSub(mul(A4,xGS),b4)) < 1e-3);
    std::cout << "Iterative OK\n";
    
    // Eigen
    Matrix A5(2,2); A5.a = {{3,1},{1,2}};
    Vector v(2,1.0);
    double lambda = powerMethod(A5,v,1000,1e-6);
    assert(approx(lambda,3.618034,1e-4));
    Matrix A6(2,2); A6.a = {{4,1},{1,2}};
    auto vals = qrEigenvalues(A6,1000,1e-6);
    assert((approx(vals[0],4.414214,1e-3) || approx(vals[1],4.414214,1e-3)));
    std::cout << "Eigen OK\n";
    
    // Misc
    Matrix A7(2,2); A7.a = {{4,7},{2,6}};
    assert(approx(determinant(A7),10.0));
    Matrix inv = inverseMatrix(A7);
    Matrix prod = mul(A7,inv);
    assert(approx(prod.a[0][0],1.0) && approx(prod.a[1][1],1.0));
    Matrix B(3,3); B.a = {{4,2,1},{2,5,3},{1,3,6}};
    Matrix Lc; assert(choleskyDecompose(B,Lc));
    Vector bc = {1,2,3};
    Vector xc = solveCholesky(Lc,bc);
    assert(vectorNormMax(vecSub(mul(B,xc),bc)) < 1e-5);
    Matrix C(3,3); C.a = {{1,2,3},{2,4,6},{1,1,1}};
    assert(matrixRank(C) == 2);
    std::cout << "Misc OK\n";
    
    std::cout << "\nAll tests passed!\n";
    return 0;
}