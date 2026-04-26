#include "GaussSolver.h"

Vector solveGauss(const Matrix &Aorig, const Vector &borig, bool usePivot) {
    int n = Aorig.n;
    assert(Aorig.n == Aorig.m && (int)borig.size() == n);
    
    Matrix A = Aorig;
    Vector b = borig;
    Vector x(n, 0.0);
    
    for (int k = 0; k < n; k++) {
        if (usePivot) {
            int maxRow = k;
            for (int i = k + 1; i < n; i++)
                if (std::abs(A.a[i][k]) > std::abs(A.a[maxRow][k]))
                    maxRow = i;
            std::swap(A.a[k], A.a[maxRow]);
            std::swap(b[k], b[maxRow]);
        }
        
        double pivot = A.a[k][k];
        // Порог снижен для плохо обусловленных матриц (Гильберта)
        if (std::abs(pivot) < 1e-15) {
            std::cerr << "Warning: very small pivot at row " << k 
                      << " (value=" << pivot << ")\n";
        }
        
        // Нормируем текущую строку
        for (int j = k; j < n; j++) A.a[k][j] /= pivot;
        b[k] /= pivot;
        
        // Обнуляем элементы ниже диагонали
        for (int i = k + 1; i < n; i++) {
            double factor = A.a[i][k];
            for (int j = k; j < n; j++) A.a[i][j] -= factor * A.a[k][j];
            b[i] -= factor * b[k];
        }
    }
    
    // Обратный ход
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) x[i] -= A.a[i][j] * x[j];
    }
    
    return x;
}