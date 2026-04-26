#include "QRDecomposition.h"

void decomposeQR(const Matrix &A, Matrix &Q, Matrix &R) {
    int n = A.n;
    assert(A.n == A.m);
    
    Q = Matrix(n, n);
    R = Matrix(n, n);
    
    // Копируем столбцы A
    std::vector<Vector> u(n, Vector(n));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            u[j][i] = A.a[i][j];
    
    std::vector<Vector> q(n, Vector(n));
    
    for (int k = 0; k < n; k++) {
        Vector v = u[k];
        
        // Вычитаем проекции на предыдущие q
        for (int j = 0; j < k; j++) {
            double dot = 0.0;
            for (int i = 0; i < n; i++)
                dot += v[i] * q[j][i];
            R.a[j][k] = dot;
            for (int i = 0; i < n; i++)
                v[i] -= dot * q[j][i];
        }
        
        // Нормируем
        double norm = 0.0;
        for (int i = 0; i < n; i++)
            norm += v[i] * v[i];
        norm = std::sqrt(norm);
        
        R.a[k][k] = norm;
        
        if (norm < 1e-12) continue; // Вырожденный случай
        
        for (int i = 0; i < n; i++)
            q[k][i] = v[i] / norm;
    }
    
    // Заполняем Q
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            Q.a[i][j] = q[j][i];
}