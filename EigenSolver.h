#ifndef EIGEN_SOLVER_H
#define EIGEN_SOLVER_H

#include "MatrixUtils.h"

// Степенной метод: находит максимальное по модулю собственное значение
double powerMethod(const Matrix &A, Vector &v, int maxIter, double tol);

// QR-алгоритм: находит все собственные значения (приближенно)
std::vector<double> qrEigenvalues(const Matrix &A, int maxIter, double tol);

#endif