#ifndef MISC_MATH_H
#define MISC_MATH_H

#include "MatrixUtils.h"

// Обратная матрица (через LU)
Matrix inverseMatrix(const Matrix &A);

// Определитель (через LU)
double determinant(const Matrix &A);

// Число обусловленности по норме ||.||_inf
double conditionNumber(const Matrix &A);

// Матрица Гильберта (для тестов)
Matrix hilbertMatrix(int n);

// Разложение Холецкого: A = L * L^T (только для симметричных положительно определенных)
// Возвращает false, если разложение невозможно
bool choleskyDecompose(const Matrix &A, Matrix &L);

// Решение системы с помощью разложения Холецкого
Vector solveCholesky(const Matrix &L, const Vector &b);

// Ранг матрицы (через приведение к ступенчатому виду)
int matrixRank(const Matrix &A, double eps = 1e-12);

#endif