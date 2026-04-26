#ifndef ITERATIVE_SOLVER_H
#define ITERATIVE_SOLVER_H

#include "MatrixUtils.h"

Vector solveJacobi(const Matrix &A, const Vector &b, const Vector &x0, double tol, int maxIter);
Vector solveGaussSeidel(const Matrix &A, const Vector &b, const Vector &x0, double tol, int maxIter);

#endif