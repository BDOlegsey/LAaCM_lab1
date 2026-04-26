#include "EigenSolver.h"
#include "QRDecomposition.h"

double powerMethod(const Matrix &A, Vector &v, int maxIter, double tol) {
    int n = A.n;
    Vector b = v;
    
    for (int it = 0; it < maxIter; it++) {
        Vector w = mul(A, b);
        
        // Нормировка по максимуму модуля
        double normw = vectorNormMax(w);
        if (normw < 1e-12) break;
        
        for (int i = 0; i < n; i++)
            w[i] /= normw;
        
        if (vectorNormMax(vecSub(w, b)) < tol) {
            b = w;
            break;
        }
        
        b = w;
    }
    
    // Отношение Релея для уточнения lambda
    Vector Ab = mul(A, b);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; i++) {
        num += b[i] * Ab[i];
        den += b[i] * b[i];
    }
    
    v = b;
    return num / den;
}

std::vector<double> qrEigenvalues(const Matrix &Aorig, int maxIter, double tol) {
    Matrix A = Aorig;
    int n = A.n;
    
    for (int it = 0; it < maxIter; it++) {
        Matrix Q, R;
        decomposeQR(A, Q, R);
        
        // A_new = R * Q
        A = mul(R, Q);
        
        // Проверяем сходимость: элементы ниже диагонали должны быть ~0
        bool converged = true;
        for (int i = 1; i < n && converged; i++)
            for (int j = 0; j < i && converged; j++)
                if (std::abs(A.a[i][j]) > tol)
                    converged = false;
        
        if (converged) break;
    }
    
    std::vector<double> lambda(n);
    for (int i = 0; i < n; i++)
        lambda[i] = A.a[i][i];
    
    return lambda;
}