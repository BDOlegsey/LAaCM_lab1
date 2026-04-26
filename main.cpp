#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>
#include "MatrixUtils.h"
#include "GaussSolver.h"
#include "LUDecomposition.h"
#include "MiscMath.h"

using namespace std;

void printUsage() {
    cout << "Usage: ./lin_algebra <task> [args]\n";
    cout << "Tasks:\n";
    cout << "  gauss-nopivot <A.txt> <b.txt>   - Gauss without pivoting\n";
    cout << "  gauss         <A.txt> <b.txt>   - Gauss with partial pivoting\n";
    cout << "  lu            <A.txt> <b.txt>   - LU decomposition\n";
    cout << "  hilbert       <n>               - Accuracy test on Hilbert matrix\n";
    cout << "  multi         <n> <k>           - Multiple RHS test (n=size, k=count)\n";
    cout << "  auto                            - Run all experiments from lab\n";
}

pair<Matrix, Vector> readSystem(const string& matFile, const string& rhsFile) {
    Matrix A;
    ifstream finA(matFile);
    if (!finA) { cerr << "Cannot open " << matFile << "\n"; exit(1); }
    A.read(finA);
    
    ifstream finB(rhsFile);
    if (!finB) { cerr << "Cannot open " << rhsFile << "\n"; exit(1); }
    Vector b = readVector(finB);
    
    if ((int)b.size() != A.n) {
        cerr << "Size mismatch\n"; exit(1);
    }
    return {A, b};
}

void printResult(const Vector& x, const Matrix& A, const Vector& b, double timeSec) {
    cout << fixed << setprecision(6);
    cout << "Time: " << timeSec << " sec\n";
    cout << "Solution:\n";
    printVector(x, cout);
    
    Vector res = vecSub(mul(A, x), b);
    cout << "Residual ||Ax-b||_inf: " << vectorNormMax(res) << "\n";
}

// Эксперимент 4.3: матрица Гильберта
void runHilbert(int n) {
    cout << "\n=== Hilbert matrix, n = " << n << " ===\n";
    Matrix H = hilbertMatrix(n);
    Vector xExact(n, 1.0);
    Vector b = mul(H, xExact);
    
    // Gauss without pivot
    auto t1 = chrono::high_resolution_clock::now();
    Vector x1 = solveGauss(H, b, false);
    auto t2 = chrono::high_resolution_clock::now();
    double time1 = chrono::duration<double>(t2 - t1).count();
    Vector err1 = vecSub(x1, xExact);
    double relErr1 = vectorNormMax(err1) / vectorNormMax(xExact);
    double res1 = vectorNormMax(vecSub(mul(H, x1), b));
    
    // Gauss with pivot
    t1 = chrono::high_resolution_clock::now();
    Vector x2 = solveGauss(H, b, true);
    t2 = chrono::high_resolution_clock::now();
    double time2 = chrono::duration<double>(t2 - t1).count();
    Vector err2 = vecSub(x2, xExact);
    double relErr2 = vectorNormMax(err2) / vectorNormMax(xExact);
    double res2 = vectorNormMax(vecSub(mul(H, x2), b));
    
    // LU
    t1 = chrono::high_resolution_clock::now();
    Matrix L, U; vector<int> P; int sw;
    decomposeLU(H, L, U, P, sw);
    Vector x3 = solveLU(L, U, P, b);
    t2 = chrono::high_resolution_clock::now();
    double time3 = chrono::duration<double>(t2 - t1).count();
    Vector err3 = vecSub(x3, xExact);
    double relErr3 = vectorNormMax(err3) / vectorNormMax(xExact);
    double res3 = vectorNormMax(vecSub(mul(H, x3), b));
    
    cout << fixed << setprecision(2);
    cout << "Method               Time(ms)  Rel.Error      Residual\n";
    cout << "Gauss (no pivot)     " << setw(8) << time1*1000 
         << "  " << scientific << relErr1 << "  " << res1 << "\n";
    cout << "Gauss (with pivot)   " << setw(8) << time2*1000 
         << "  " << relErr2 << "  " << res2 << "\n";
    cout << "LU decomposition     " << setw(8) << time3*1000 
         << "  " << relErr3 << "  " << res3 << "\n";
}

// Эксперимент 4.2: множественные правые части
void runMultipleRHS(int n, int k) {
    cout << "\n=== Multiple RHS, n = " << n << ", k = " << k << " ===\n";
    
    // Генерация случайной матрицы с фиксированным seed
    srand(42);
    Matrix A(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A.a[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    
    vector<Vector> rhsList(k, Vector(n));
    for (int r = 0; r < k; r++)
        for (int i = 0; i < n; i++)
            rhsList[r][i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    
    // Метод Гаусса с выбором: решаем каждую систему заново
    auto t1 = chrono::high_resolution_clock::now();
    for (int r = 0; r < k; r++) {
        Vector x = solveGauss(A, rhsList[r], true);
    }
    auto t2 = chrono::high_resolution_clock::now();
    double timeGauss = chrono::duration<double>(t2 - t1).count();
    
    // LU: одно разложение + k подстановок
    t1 = chrono::high_resolution_clock::now();
    Matrix L, U; vector<int> P; int sw;
    decomposeLU(A, L, U, P, sw);
    auto tLUdecomp = chrono::high_resolution_clock::now();
    for (int r = 0; r < k; r++) {
        Vector x = solveLU(L, U, P, rhsList[r]);
    }
    t2 = chrono::high_resolution_clock::now();
    double timeLUtotal = chrono::duration<double>(t2 - t1).count();
    double timeLUdecomp = chrono::duration<double>(tLUdecomp - t1).count();
    
    cout << fixed << setprecision(3);
    cout << "Gauss (k times):     " << timeGauss*1000 << " ms\n";
    cout << "LU (decomp + solves): " << timeLUtotal*1000 << " ms";
    cout << "  (decomp: " << timeLUdecomp*1000 << " ms, solves: " 
         << (timeLUtotal - timeLUdecomp)*1000 << " ms)\n";
    if (timeLUtotal > 0)
        cout << "Speedup: " << timeGauss / timeLUtotal << "x\n";
}

// Эксперимент 4.1: сравнение времени одного решения
void runTimingComparison() {
    cout << "\n=== Timing comparison (single system) ===\n";
    vector<int> sizes = {100, 200, 500, 1000};
    
    cout << "   n    Gauss(no)   Gauss(piv)   LU(total)   LU(decomp)   LU(solve)\n";
    for (int n : sizes) {
        srand(42);
        Matrix A(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A.a[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Vector b(n);
        for (int i = 0; i < n; i++) b[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        
        auto t1 = chrono::high_resolution_clock::now();
        solveGauss(A, b, false);
        auto t2 = chrono::high_resolution_clock::now();
        double t_nopiv = chrono::duration<double>(t2-t1).count();
        
        t1 = chrono::high_resolution_clock::now();
        solveGauss(A, b, true);
        t2 = chrono::high_resolution_clock::now();
        double t_piv = chrono::duration<double>(t2-t1).count();
        
        t1 = chrono::high_resolution_clock::now();
        Matrix L, U; vector<int> P; int sw;
        decomposeLU(A, L, U, P, sw);
        auto tLU = chrono::high_resolution_clock::now();
        solveLU(L, U, P, b);
        t2 = chrono::high_resolution_clock::now();
        double t_lu_total = chrono::duration<double>(t2-t1).count();
        double t_lu_decomp = chrono::duration<double>(tLU-t1).count();
        
        cout << setw(5) << n << setprecision(3) << fixed
             << setw(12) << t_nopiv*1000 << " ms"
             << setw(12) << t_piv*1000 << " ms"
             << setw(12) << t_lu_total*1000 << " ms"
             << setw(13) << t_lu_decomp*1000 << " ms"
             << setw(12) << (t_lu_total - t_lu_decomp)*1000 << " ms\n";
    }
}

void runAutoExperiments() {
    runTimingComparison();
    runMultipleRHS(500, 1);
    runMultipleRHS(500, 10);
    runMultipleRHS(500, 100);
    for (int n : {5, 10, 15}) runHilbert(n);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(); return 1; }
    
    string task = argv[1];
    
    if (task == "gauss-nopivot" && argc == 4) {
        auto [A, b] = readSystem(argv[2], argv[3]);
        auto t1 = chrono::high_resolution_clock::now();
        Vector x = solveGauss(A, b, false);
        auto t2 = chrono::high_resolution_clock::now();
        printResult(x, A, b, chrono::duration<double>(t2-t1).count());
    }
    else if (task == "gauss" && argc == 4) {
        auto [A, b] = readSystem(argv[2], argv[3]);
        auto t1 = chrono::high_resolution_clock::now();
        Vector x = solveGauss(A, b, true);
        auto t2 = chrono::high_resolution_clock::now();
        printResult(x, A, b, chrono::duration<double>(t2-t1).count());
    }
    else if (task == "lu" && argc == 4) {
        auto [A, b] = readSystem(argv[2], argv[3]);
        auto t1 = chrono::high_resolution_clock::now();
        Matrix L, U; vector<int> P; int sw;
        decomposeLU(A, L, U, P, sw);
        Vector x = solveLU(L, U, P, b);
        auto t2 = chrono::high_resolution_clock::now();
        printResult(x, A, b, chrono::duration<double>(t2-t1).count());
    }
    else if (task == "hilbert" && argc == 3) {
        runHilbert(stoi(argv[2]));
    }
    else if (task == "multi" && argc == 4) {
        runMultipleRHS(stoi(argv[2]), stoi(argv[3]));
    }
    else if (task == "auto") {
        runAutoExperiments();
    }
    else {
        printUsage();
        return 1;
    }
    return 0;
}