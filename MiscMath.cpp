#include "MiscMath.h"
#include "LUDecomposition.h"

Matrix inverseMatrix(const Matrix &A) {
    int n = A.n;
    assert(A.n == A.m);
    
    Matrix inv(n, n);
    Matrix L, U;
    std::vector<int> P;
    int swaps;
    decomposeLU(A, L, U, P, swaps);
    
    for (int col = 0; col < n; col++) {
        Vector e(n, 0.0);
        e[col] = 1.0;
        Vector x = solveLU(L, U, P, e);
        for (int row = 0; row < n; row++)
            inv.a[row][col] = x[row];
    }
    
    return inv;
}

double determinant(const Matrix &A) {
    int n = A.n;
    Matrix L, U;
    std::vector<int> P;
    int swapCount;
    decomposeLU(A, L, U, P, swapCount);
    
    double det = 1.0;
    for (int i = 0; i < n; i++)
        det *= U.a[i][i];
    
    // Учитываем знак перестановок
    if (swapCount % 2 != 0)
        det = -det;
    
    return det;
}

double conditionNumber(const Matrix &A) {
    int n = A.n;
    assert(A.n == A.m);
    
    double normA = matrixNormInf(A);
    Matrix invA = inverseMatrix(A);
    double normInvA = matrixNormInf(invA);
    
    return normA * normInvA;
}

Matrix hilbertMatrix(int n) {
    Matrix H(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            H.a[i][j] = 1.0 / (i + j + 1.0);
    return H;
}

bool choleskyDecompose(const Matrix &A, Matrix &L) {
    int n = A.n;
    assert(A.n == A.m);
    L = Matrix(n, n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += L.a[i][k] * L.a[j][k];
            
            if (i == j) {
                double val = A.a[i][i] - sum;
                if (val <= 0) {
                    std::cerr << "Error: matrix is not positive definite.\n";
                    return false;
                }
                L.a[i][i] = std::sqrt(val);
            } else {
                if (std::abs(L.a[j][j]) < 1e-12)
                    return false;
                L.a[i][j] = (A.a[i][j] - sum) / L.a[j][j];
            }
        }
    }
    return true;
}

Vector solveCholesky(const Matrix &L, const Vector &b) {
    int n = L.n;
    Vector y(n, 0.0), x(n, 0.0);
    
    // Прямая подстановка: L * y = b
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++)
            sum += L.a[i][j] * y[j];
        y[i] = (b[i] - sum) / L.a[i][i];
    }
    
    // Обратная подстановка: L^T * x = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++)
            sum += L.a[j][i] * x[j]; // L^T[i][j] = L[j][i]
        x[i] = (y[i] - sum) / L.a[i][i];
    }
    
    return x;
}

int matrixRank(const Matrix &A, double eps) {
    int n = A.n;
    int m = A.m;
    Matrix B = A;
    int rank = 0;
    
    for (int col = 0, row = 0; col < m && row < n; col++) {
        // Ищем ведущий элемент
        int pivotRow = row;
        double maxVal = std::abs(B.a[row][col]);
        for (int i = row + 1; i < n; i++) {
            if (std::abs(B.a[i][col]) > maxVal) {
                maxVal = std::abs(B.a[i][col]);
                pivotRow = i;
            }
        }
        
        if (maxVal < eps) continue; // Столбец линейно зависим
        
        std::swap(B.a[row], B.a[pivotRow]);
        
        // Нормируем и обнуляем
        double pivot = B.a[row][col];
        for (int j = col; j < m; j++)
            B.a[row][j] /= pivot;
        
        for (int i = 0; i < n; i++) {
            if (i != row && std::abs(B.a[i][col]) > eps) {
                double factor = B.a[i][col];
                for (int j = col; j < m; j++)
                    B.a[i][j] -= factor * B.a[row][j];
            }
        }
        
        row++;
        rank++;
    }
    
    return rank;
}