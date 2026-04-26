#ifndef LU_DECOMPOSITION_H
#define LU_DECOMPOSITION_H

#include "MatrixUtils.h"

// LU-разложение с частичным выбором главного элемента
// Возвращает L, U и вектор перестановок P, а также число перестановок (для знака определителя)
void decomposeLU(const Matrix &A, Matrix &L, Matrix &U, std::vector<int> &P, int &swapCount);
Vector solveLU(const Matrix &L, const Matrix &U, const std::vector<int> &P, const Vector &b);

#endif