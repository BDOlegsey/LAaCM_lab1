#include "LUDecomposition.h"

void decomposeLU(const Matrix &Aorig, Matrix &L, Matrix &U, std::vector<int> &P, int &swapCount) {
    int n = Aorig.n;
    assert(Aorig.n == Aorig.m);
    
    Matrix A = Aorig;
    L = Matrix(n, n);
    U = Matrix(n, n);
    P.resize(n);
    swapCount = 0;
    
    for (int i = 0; i < n; i++) 
        P[i] = i;
    
    for (int k = 0; k < n; k++) {
        // Частичный выбор главного элемента по столбцу
        int maxRow = k;
        double maxVal = std::abs(A.a[k][k]);
        for (int i = k + 1; i < n; i++) {
            if (std::abs(A.a[i][k]) > maxVal) {
                maxVal = std::abs(A.a[i][k]);
                maxRow = i;
            }
        }
        
        if (maxRow != k) {
            std::swap(A.a[k], A.a[maxRow]);
            std::swap(P[k], P[maxRow]);
            swapCount++;
        }
        
        double pivot = A.a[k][k];
        if (std::abs(pivot) < 1e-12) {
            std::cerr << "Error: matrix is singular or nearly singular.\n";
            break;
        }
        
        L.a[k][k] = 1.0;
        U.a[k][k] = pivot;
        
        // Заполняем k-ю строку U и k-й столбец L
        for (int j = k + 1; j < n; j++) {
            U.a[k][j] = A.a[k][j];
            L.a[j][k] = A.a[j][k] / pivot;
        }
        
        // Обновляем оставшуюся подматрицу (дополнение Шура)
        for (int i = k + 1; i < n; i++)
            for (int j = k + 1; j < n; j++)
                A.a[i][j] -= L.a[i][k] * U.a[k][j];
    }
}

Vector solveLU(const Matrix &L, const Matrix &U, const std::vector<int> &P, const Vector &b) {
    int n = L.n;
    Vector x(n, 0.0), y(n, 0.0), bp(n);
    
    // Применяем перестановку к вектору b: bp = P*b
    for (int i = 0; i < n; i++) 
        bp[i] = b[P[i]];
    
    // Прямая подстановка: L * y = bp
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++)
            sum += L.a[i][j] * y[j];
        y[i] = bp[i] - sum;
    }
    
    // Обратная подстановка: U * x = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++)
            sum += U.a[i][j] * x[j];
        x[i] = (y[i] - sum) / U.a[i][i];
    }
    
    return x;
}