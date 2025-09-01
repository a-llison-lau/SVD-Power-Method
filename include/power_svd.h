#ifndef POWER_SVD_H
#define POWER_SVD_H

#include <vector>
#include <tuple>

class PowerSVD {
public:
    PowerSVD(int max_iterations = 1000, double tolerance = 1e-10);
    
    // Compute dominant singular value and vectors using power method
    std::tuple<double, std::vector<double>, std::vector<double>> 
    compute_dominant_svd(const std::vector<std::vector<double>>& matrix);
    
    // Compute top k singular values and vectors
    std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    compute_svd(const std::vector<std::vector<double>>& matrix, int k);
    
private:
    int max_iterations_;
    double tolerance_;
    
    // Helper functions
    std::vector<double> matrix_vector_multiply(
        const std::vector<std::vector<double>>& matrix,
        const std::vector<double>& vector);
    
    std::vector<double> matrix_transpose_vector_multiply(
        const std::vector<std::vector<double>>& matrix,
        const std::vector<double>& vector);
    
    double vector_norm(const std::vector<double>& vector);
    void normalize_vector(std::vector<double>& vector);
    double dot_product(const std::vector<double>& a, const std::vector<double>& b);
    
    std::vector<std::vector<double>> deflate_matrix(
        const std::vector<std::vector<double>>& matrix,
        const std::vector<double>& u,
        const std::vector<double>& v,
        double sigma);
};

#endif