#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>

class Matrix {
public:
    int n, m;
    std::vector<std::vector<double>> a;
    
    Matrix(int n_, int m_) : n(n_), m(m_), a(n_, std::vector<double>(m_, 0.0)) {}
    Matrix() : n(0), m(0) {}
    
    void read(std::istream &in) {
        in >> n >> m;
        a.assign(n, std::vector<double>(m));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                in >> a[i][j];
    }
    
    void print(std::ostream &out) const {
        out << n << " " << m << "\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++)
                out << a[i][j] << " ";
            out << "\n";
        }
    }
    
    std::vector<double>& operator[](int i) { return a[i]; }
    const std::vector<double>& operator[](int i) const { return a[i]; }
};

using Vector = std::vector<double>;

inline Vector readVector(std::istream &in) {
    int n; 
    in >> n;
    Vector v(n);
    for (int i = 0; i < n; i++) 
        in >> v[i];
    return v;
}

inline void printVector(const Vector &v, std::ostream &out) {
    int n = v.size();
    out << n << "\n";
    for (int i = 0; i < n; i++) 
        out << v[i] << "\n";
}

inline Vector mul(const Matrix &A, const Vector &x) {
    assert(A.m == (int)x.size());
    Vector y(A.n, 0.0);
    for (int i = 0; i < A.n; i++)
        for (int j = 0; j < A.m; j++)
            y[i] += A.a[i][j] * x[j];
    return y;
}

inline Matrix mul(const Matrix &A, const Matrix &B) {
    assert(A.m == B.n);
    Matrix C(A.n, B.m);
    for (int i = 0; i < A.n; i++)
        for (int j = 0; j < B.m; j++) {
            double sum = 0.0;
            for (int k = 0; k < A.m; k++)
                sum += A.a[i][k] * B.a[k][j];
            C.a[i][j] = sum;
        }
    return C;
}

inline Matrix transpose(const Matrix &A) {
    Matrix B(A.m, A.n);
    for (int i = 0; i < A.n; i++)
        for (int j = 0; j < A.m; j++)
            B.a[j][i] = A.a[i][j];
    return B;
}

inline double vectorNormMax(const Vector &x) {
    double maxv = 0.0;
    for (double xi : x)
        maxv = std::max(maxv, std::abs(xi));
    return maxv;
}

inline double vectorNorm1(const Vector &x) {
    double sum = 0.0;
    for (double xi : x)
        sum += std::abs(xi);
    return sum;
}

inline double matrixNormInf(const Matrix &A) {
    double norm = 0.0;
    for (int i = 0; i < A.n; i++) {
        double rowSum = 0.0;
        for (int j = 0; j < A.m; j++)
            rowSum += std::abs(A.a[i][j]);
        norm = std::max(norm, rowSum);
    }
    return norm;
}

inline double matrixNorm1(const Matrix &A) {
    double norm = 0.0;
    for (int j = 0; j < A.m; j++) {
        double colSum = 0.0;
        for (int i = 0; i < A.n; i++)
            colSum += std::abs(A.a[i][j]);
        norm = std::max(norm, colSum);
    }
    return norm;
}

inline Vector vecSub(const Vector &a, const Vector &b) {
    assert(a.size() == b.size());
    Vector res(a.size());
    for (size_t i = 0; i < a.size(); i++)
        res[i] = a[i] - b[i];
    return res;
}

#endif