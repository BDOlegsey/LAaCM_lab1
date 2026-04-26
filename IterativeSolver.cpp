#include "IterativeSolver.h"

Vector solveJacobi(const Matrix &A, const Vector &b, const Vector &x0, double tol, int maxIter) {
    int n = A.n;
    Vector x = x0;
    Vector xNew(n, 0.0);
    
    for (int iter = 0; iter < maxIter; iter++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j)
                    sum += A.a[i][j] * x[j];
            }
            xNew[i] = (b[i] - sum) / A.a[i][i];
        }
        
        if (vectorNormMax(vecSub(xNew, x)) < tol)
            return xNew;
        
        x = xNew;
    }
    
    return x;
}

Vector solveGaussSeidel(const Matrix &A, const Vector &b, const Vector &x0, double tol, int maxIter) {
    int n = A.n;
    Vector x = x0;
    
    for (int iter = 0; iter < maxIter; iter++) {
        Vector old = x;
        
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j)
                    sum += A.a[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / A.a[i][i];
        }
        
        if (vectorNormMax(vecSub(x, old)) < tol)
            return x;
    }
    
    return x;
}