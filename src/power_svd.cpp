#include "power_svd.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

PowerSVD::PowerSVD(int max_iterations, double tolerance) 
    : max_iterations_(max_iterations), tolerance_(tolerance) {}

std::vector<double> PowerSVD::matrix_vector_multiply(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>& vector) {
    
    int m = matrix.size();
    int n = matrix[0].size();
    
    if (vector.size() != n) {
        throw std::invalid_argument("Matrix-vector dimensions don't match");
    }
    
    std::vector<double> result(m, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

std::vector<double> PowerSVD::matrix_transpose_vector_multiply(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>& vector) {
    
    int m = matrix.size();
    int n = matrix[0].size();
    
    if (vector.size() != m) {
        throw std::invalid_argument("Matrix transpose-vector dimensions don't match");
    }
    
    std::vector<double> result(n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            result[j] += matrix[i][j] * vector[i];
        }
    }
    return result;
}

double PowerSVD::vector_norm(const std::vector<double>& vector) {
    double norm = 0.0;
    for (double val : vector) {
        norm += val * val;
    }
    return std::sqrt(norm);
}

void PowerSVD::normalize_vector(std::vector<double>& vector) {
    double norm = vector_norm(vector);
    if (norm > 0) {
        for (double& val : vector) {
            val /= norm;
        }
    }
}

double PowerSVD::dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

std::tuple<double, std::vector<double>, std::vector<double>> 
PowerSVD::compute_dominant_svd(const std::vector<std::vector<double>>& matrix) {
    
    int m = matrix.size();
    int n = matrix[0].size();
    
    // Initialize random right singular vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);
    
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = d(gen);
    }
    normalize_vector(v);
    
    std::vector<double> u(m);
    double sigma = 0.0;
    double prev_sigma = 0.0;
    
    for (int iter = 0; iter < max_iterations_; ++iter) {
        // u = A * v
        u = matrix_vector_multiply(matrix, v);
        sigma = vector_norm(u);
        
        if (sigma > 0) {
            normalize_vector(u);
        }
        
        // v = A^T * u
        v = matrix_transpose_vector_multiply(matrix, u);
        normalize_vector(v);
        
        // Check convergence
        if (iter > 0 && std::abs(sigma - prev_sigma) < tolerance_) {
            break;
        }
        prev_sigma = sigma;
    }
    
    return std::make_tuple(sigma, u, v);
}

std::vector<std::vector<double>> PowerSVD::deflate_matrix(
    const std::vector<std::vector<double>>& matrix,
    const std::vector<double>& u,
    const std::vector<double>& v,
    double sigma) {
    
    int m = matrix.size();
    int n = matrix[0].size();
    
    std::vector<std::vector<double>> deflated(m, std::vector<double>(n));
    
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            deflated[i][j] = matrix[i][j] - sigma * u[i] * v[j];
        }
    }
    
    return deflated;
}

std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
PowerSVD::compute_svd(const std::vector<std::vector<double>>& matrix, int k) {
    
    std::vector<double> singular_values;
    std::vector<std::vector<double>> left_vectors;
    std::vector<std::vector<double>> right_vectors;
    
    auto current_matrix = matrix;
    
    for (int i = 0; i < k; ++i) {
        auto [sigma, u, v] = compute_dominant_svd(current_matrix);
        
        if (sigma < tolerance_) {
            break; // Matrix is effectively rank-deficient
        }
        
        singular_values.push_back(sigma);
        left_vectors.push_back(u);
        right_vectors.push_back(v);
        
        // Deflate the matrix
        current_matrix = deflate_matrix(current_matrix, u, v, sigma);
    }
    
    return std::make_tuple(singular_values, left_vectors, right_vectors);
}

// Python bindings
namespace py = pybind11;

PYBIND11_MODULE(power_svd_cpp, m) {
    m.doc() = "Power method SVD implementation in C++";
    
    py::class_<PowerSVD>(m, "PowerSVD")
        .def(py::init<int, double>(), 
             py::arg("max_iterations") = 1000, 
             py::arg("tolerance") = 1e-10)
        .def("compute_dominant_svd", &PowerSVD::compute_dominant_svd,
             "Compute the dominant singular value and vectors")
        .def("compute_svd", &PowerSVD::compute_svd,
             "Compute top k singular values and vectors",
             py::arg("matrix"), py::arg("k"));
}