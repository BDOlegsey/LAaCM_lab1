#ifndef QR_DECOMPOSITION_H
#define QR_DECOMPOSITION_H

#include "MatrixUtils.h"

// QR-разложение методом Грамма-Шмидтаы
void decomposeQR(const Matrix &A, Matrix &Q, Matrix &R);

#endif