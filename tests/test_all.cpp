#include <iostream>
#include <cassert>
#include <cmath>
#include "../MatrixUtils.h"
#include "../GaussSolver.h"
#include "../LUDecomposition.h"
#include "../MiscMath.h"

bool approx(double a, double b, double eps = 1e-5) {
    return std::abs(a - b) < eps;
}

int main() {
    std::cout << "Running tests...\n";
    
    // Test 1: Gauss without pivot
    Matrix A1(2,2); A1.a = {{2,1},{1,3}};
    Vector b1 = {5,6};
    Vector x1 = solveGauss(A1, b1, false);
    assert(approx(x1[0], 1.8) && approx(x1[1], 1.4));
    std::cout << "Gauss (no pivot) OK\n";
    
    // Test 2: Gauss with pivot
    Vector x2 = solveGauss(A1, b1, true);
    assert(approx(x2[0], 1.8) && approx(x2[1], 1.4));
    std::cout << "Gauss (with pivot) OK\n";
    
    // Test 3: LU decomposition
    Matrix A3(3,3); A3.a = {{4,3,2},{6,3,1},{2,1,3}};
    Vector b3 = {1,2,3};
    Matrix L,U; std::vector<int> P; int sw;
    decomposeLU(A3, L, U, P, sw);
    Vector x3 = solveLU(L, U, P, b3);
    assert(vectorNormMax(vecSub(mul(A3,x3), b3)) < 1e-5);
    std::cout << "LU OK\n";
    
    // Test 4: Hilbert matrix
    Matrix H = hilbertMatrix(4);
    Vector xExact(4, 1.0);
    Vector bH = mul(H, xExact);
    Vector xH = solveGauss(H, bH, true);
    assert(vectorNormMax(vecSub(xH, xExact)) < 1e-3);
    std::cout << "Hilbert OK\n";
    
    // Test 5: Determinant
    Matrix A5(2,2); A5.a = {{4,7},{2,6}};
    assert(approx(determinant(A5), 10.0));
    std::cout << "Determinant OK\n";
    
    std::cout << "\nAll tests passed!\n";
    return 0;
}