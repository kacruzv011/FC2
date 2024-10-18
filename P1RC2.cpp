#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <limits>

// Función para resolver por descomposición LU
Eigen::VectorXd solve_LU(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(A);
    return lu_decomp.solve(b);
}

// Función para resolver por Eigen usando QR
Eigen::VectorXd solve_QR(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr_decomp(A);
    return qr_decomp.solve(b);
}

// Función para resolver por Gauss-Jordan con pivotación
Eigen::VectorXd solve_Gauss_Jordan(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::MatrixXd augmented(A.rows(), A.cols() + 1);
    augmented << A, b;

    for (int i = 0; i < A.rows(); i++) {
        double maxPivot = std::abs(augmented(i, i));
        int maxRow = i;
        for (int k = i + 1; k < A.rows(); k++) {
            if (std::abs(augmented(k, i)) > maxPivot) {
                maxPivot = std::abs(augmented(k, i));
                maxRow = k;
            }
        }

        if (maxRow != i) {
            augmented.row(i).swap(augmented.row(maxRow));
        }

        double pivot = augmented(i, i);
        if (std::abs(pivot) < 1e-10) {
            augmented(i, i) += 1e-5;
            pivot = augmented(i, i);
        }

        for (int j = 0; j < augmented.cols(); j++) {
            augmented(i, j) /= pivot;
        }

        for (int j = 0; j < A.rows(); j++) {
            if (j != i) {
                double factor = augmented(j, i);
                for (int k = 0; k < augmented.cols(); k++) {
                    augmented(j, k) -= factor * augmented(i, k);
                }
            }
        }
    }

    return augmented.col(augmented.cols() - 1);
}

// Construcción de la matriz RCRC infinita
Eigen::MatrixXd construct_RCRC_infinite_matrix(double R, double C, int N) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, N);
    for (int i = 0; i < N; i++) {
        A(i, i) = 1.0 / (R * C) + 1e-5;
        if (i > 0) A(i, i - 1) = -1.0 / (R * C);
        if (i < N - 1) A(i, i + 1) = -1.0 / (R * C);
    }
    return A;
}

int main() {
    double R = 100.0;
    double C = 0.01;
    double V0 = 5.0;
    int total_iterations = 500; // Para N de 10 a 500, incrementando en 10

    std::ofstream file("times.csv");
    file << "N,LU,QR,Gauss-Jordan\n"; 

    auto overall_start = std::chrono::high_resolution_clock::now(); // Inicio del programa

    for (int step = 0; step < total_iterations; ++step) {
        int N = 10 + step * 10; // Incremento en 10
        Eigen::MatrixXd A = construct_RCRC_infinite_matrix(R, C, N);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
        b(0) = V0;

        // Medir tiempo para LU
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd solution_LU = solve_LU(A, b);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_LU = end - start;

        // Medir tiempo para QR
        start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd solution_QR = solve_QR(A, b);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_QR = end - start;

        // Medir tiempo para Gauss-Jordan
        start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd solution_GJ = solve_Gauss_Jordan(A, b);
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_GJ = end - start;

        // Guardar tiempos
        file << N << "," << elapsed_LU.count() << "," << elapsed_QR.count() << "," << elapsed_GJ.count() << "\n";

        // Calcular el tiempo transcurrido total
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_elapsed_time = current_time - overall_start;

        // Estimar el tiempo restante
        double progress = (double)(step + 1) / total_iterations;
        double estimated_total_time = total_elapsed_time.count() / progress;
        double remaining_time = estimated_total_time - total_elapsed_time.count();

        // Mostrar progreso y tiempo estimado restante
        std::cout << "Progreso: " << (step + 1) << "/" << total_iterations
                  << " (" << (progress * 100) << "%)"
                  << ", Tiempo restante estimado: " << remaining_time << " segundos.\n";
    }

    file.close();
    return 0;
}