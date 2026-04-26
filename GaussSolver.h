#ifndef GAUSS_SOLVER_H
#define GAUSS_SOLVER_H

#include "MatrixUtils.h"

Vector solveGauss(const Matrix &A, const Vector &b, bool usePivot);

#endif