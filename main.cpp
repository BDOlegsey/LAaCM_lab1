#include <iostream>
#include <fstream>
#include <chrono>
#include "MatrixUtils.h"
#include "GaussSolver.h"
#include "LUDecomposition.h"
#include "QRDecomposition.h"
#include "IterativeSolver.h"
#include "EigenSolver.h"
#include "MiscMath.h"

void printUsage() {
    std::cout << "Usage: ./program <task> <matrix_file> [rhs_file]\n";
    std::cout << "Tasks:\n";
    std::cout << "  gauss      - Gaussian elimination with pivoting\n";
    std::cout << "  lu         - LU decomposition\n";
    std::cout << "  cholesky   - Cholesky decomposition (SPD matrices)\n";
    std::cout << "  qr         - QR decomposition\n";
    std::cout << "  jacobi     - Jacobi iterative method\n";
    std::cout << "  gs         - Gauss-Seidel iterative method\n";
    std::cout << "  power      - Power method (max eigenvalue)\n";
    std::cout << "  qr-eig     - QR algorithm for eigenvalues\n";
    std::cout << "  det        - Determinant\n";
    std::cout << "  inv        - Inverse matrix\n";
    std::cout << "  cond       - Condition number\n";
    std::cout << "  rank       - Matrix rank\n";
    std::cout << "\nFormat of matrix_file: n m followed by n*m numbers\n";
    std::cout << "Format of rhs_file: n followed by n numbers\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage();
        return 1;
    }
    
    std::string task = argv[1];
    std::string matFile = argv[2];
    
    Matrix A;
    Vector b;
    
    // Читаем матрицу
    std::ifstream finA(matFile);
    if (!finA) {
        std::cerr << "Error: cannot open matrix file " << matFile << "\n";
        return 1;
    }
    A.read(finA);
    
    // Для задач с правой частью читаем вектор b
    bool needRhs = (task == "gauss" || task == "lu" || task == "cholesky" || 
                    task == "jacobi" || task == "gs");
    if (needRhs) {
        if (argc < 4) {
            std::cerr << "Error: RHS file required for task " << task << "\n";
            return 1;
        }
        std::ifstream finB(argv[3]);
        if (!finB) {
            std::cerr << "Error: cannot open RHS file " << argv[3] << "\n";
            return 1;
        }
        b = readVector(finB);
        if ((int)b.size() != A.n) {
            std::cerr << "Error: size mismatch between matrix and RHS\n";
            return 1;
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (task == "gauss") {
        Vector x = solveGauss(A, b, true);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        std::cout << "Time: " << diff.count() << " sec\n";
        std::cout << "Solution:\n";
        printVector(x, std::cout);
        
        Vector res = vecSub(mul(A, x), b);
        std::cout << "Residual norm (inf): " << vectorNormMax(res) << "\n";
    }
    else if (task == "lu") {
        Matrix L, U;
        std::vector<int> P;
        int swaps;
        decomposeLU(A, L, U, P, swaps);
        Vector x = solveLU(L, U, P, b);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        std::cout << "Time: " << diff.count() << " sec\n";
        std::cout << "Solution:\n";
        printVector(x, std::cout);
        
        Vector res = vecSub(mul(A, x), b);
        std::cout << "Residual norm (inf): " << vectorNormMax(res) << "\n";
    }
    else if (task == "cholesky") {
        Matrix L;
        if (!choleskyDecompose(A, L)) {
            std::cerr << "Cholesky decomposition failed.\n";
            return 1;
        }
        Vector x = solveCholesky(L, b);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        std::cout << "Time: " << diff.count() << " sec\n";
        std::cout << "Solution:\n";
        printVector(x, std::cout);
        
        Vector res = vecSub(mul(A, x), b);
        std::cout << "Residual norm (inf): " << vectorNormMax(res) << "\n";
    }
    else if (task == "qr") {
        Matrix Q, R;
        decomposeQR(A, Q, R);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        std::cout << "Time: " << diff.count() << " sec\n";
        std::cout << "Q^T * Q (should be identity):\n";
        Matrix Qt = transpose(Q);
        Matrix QtQ = mul(Qt, Q);
        QtQ.print(std::cout);
        
        std::cout << "Reconstruction error ||A - Q*R||_inf: ";
        Matrix QR = mul(Q, R);
        double err = 0.0;
        for (int i = 0; i < A.n; i++)
            for (int j = 0; j < A.m; j++)
                err = std::max(err, std::abs(A.a[i][j] - QR.a[i][j]));
        std::cout << err << "\n";
    }
    else if (task == "jacobi") {
        Vector x0(A.n, 0.0);
        Vector x = solveJacobi(A, b, x0, 1e-6, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        std::cout << "Time: " << diff.count() << " sec\n";
        std::cout << "Solution:\n";
        printVector(x, std::cout);
        
        Vector res = vecSub(mul(A, x), b);
        std::cout << "Residual norm (inf): " << vectorNormMax(res) << "\n";
    }
    else if (task == "gs") {
        Vector x0(A.n, 0.0);
        Vector x = solveGaussSeidel(A, b, x0, 1e-6, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        std::cout << "Time: " << diff.count() << " sec\n";
        std::cout << "Solution:\n";
        printVector(x, std::cout);
        
        Vector res = vecSub(mul(A, x), b);
        std::cout << "Residual norm (inf): " << vectorNormMax(res) << "\n";
    }
    else if (task == "power") {
        Vector v(A.n, 1.0);
        double lambda = powerMethod(A, v, 1000, 1e-6);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        std::cout << "Time: " << diff.count() << " sec\n";
        std::cout << "Max eigenvalue: " << lambda << "\n";
        std::cout << "Eigenvector:\n";
        printVector(v, std::cout);
        
        // Проверка: A*v ≈ lambda*v
        Vector Av = mul(A, v);
        Vector lv = v;
        for (int i = 0; i < A.n; i++) lv[i] *= lambda;
        std::cout << "Error ||Av - lambda*v||_inf: " << vectorNormMax(vecSub(Av, lv)) << "\n";
    }
    else if (task == "qr-eig") {
        std::vector<double> lambda = qrEigenvalues(A, 1000, 1e-6);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        std::cout << "Time: " << diff.count() << " sec\n";
        std::cout << "Eigenvalues:\n";
        for (double val : lambda)
            std::cout << val << "\n";
    }
    else if (task == "det") {
        double det = determinant(A);
        std::cout << "Determinant: " << det << "\n";
    }
    else if (task == "inv") {
        Matrix inv = inverseMatrix(A);
        std::cout << "Inverse matrix:\n";
        inv.print(std::cout);
        
        Matrix prod = mul(A, inv);
        std::cout << "A * A^{-1} (should be identity):\n";
        prod.print(std::cout);
    }
    else if (task == "cond") {
        double cond = conditionNumber(A);
        std::cout << "Condition number (inf-norm): " << cond << "\n";
    }
    else if (task == "rank") {
        int r = matrixRank(A);
        std::cout << "Matrix rank: " << r << "\n";
    }
    else {
        std::cerr << "Unknown task: " << task << "\n";
        printUsage();
        return 1;
    }
    
    return 0;
}