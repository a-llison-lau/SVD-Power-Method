import numpy as np
import power_svd_cpp
import time

def test_power_svd():
    # Create test matrix
    np.random.seed(42)
    A = np.random.randn(10, 8)
    
    # Convert to list of lists for C++
    A_list = A.tolist()
    
    # Initialize PowerSVD
    power_svd = power_svd_cpp.PowerSVD(max_iterations=1000, tolerance=1e-10)
    
    # Test dominant SVD
    print("Testing dominant SVD computation...")
    start_time = time.time()
    sigma, u, v = power_svd.compute_dominant_svd(A_list)
    cpp_time = time.time() - start_time
    
    print(f"C++ Power Method - Dominant singular value: {sigma:.6f}")
    print(f"C++ computation time: {cpp_time:.6f} seconds")
    
    # Compare with NumPy SVD
    start_time = time.time()
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    numpy_time = time.time() - start_time
    
    print(f"NumPy SVD - Dominant singular value: {S_np[0]:.6f}")
    print(f"NumPy computation time: {numpy_time:.6f} seconds")
    print(f"Difference: {abs(sigma - S_np[0]):.6f}")
    
    # Test top-k SVD
    print(f"\nTesting top-3 SVD computation...")
    k = min(3, min(A.shape))
    start_time = time.time()
    sigmas, U_list, V_list = power_svd.compute_svd(A_list, k)
    cpp_time_k = time.time() - start_time
    
    print(f"C++ Power Method - Top {len(sigmas)} singular values: {sigmas}")
    print(f"C++ computation time (top-{k}): {cpp_time_k:.6f} seconds")
    print(f"NumPy SVD - Top {k} singular values: {S_np[:k].tolist()}")
    
    # Verify reconstruction quality
    if len(sigmas) > 0:
        U_cpp = np.array(U_list).T  # Transpose because we store column-wise
        V_cpp = np.array(V_list)    # Already row-wise (right singular vectors)
        S_cpp = np.array(sigmas)
        
        # Reconstruct matrix
        A_reconstructed = U_cpp @ np.diag(S_cpp) @ V_cpp
        reconstruction_error = np.linalg.norm(A - A_reconstructed, 'fro')
        print(f"Reconstruction error (Frobenius norm): {reconstruction_error:.6f}")

def benchmark_different_sizes():
    print("\n" + "="*50)
    print("BENCHMARKING DIFFERENT MATRIX SIZES")
    print("="*50)
    
    sizes = [(50, 40), (100, 80), (200, 150)]
    
    for m, n in sizes:
        print(f"\nMatrix size: {m} x {n}")
        A = np.random.randn(m, n)
        A_list = A.tolist()
        
        power_svd = power_svd_cpp.PowerSVD(max_iterations=500, tolerance=1e-8)
        
        # Time C++ implementation
        start_time = time.time()
        sigma, u, v = power_svd.compute_dominant_svd(A_list)
        cpp_time = time.time() - start_time
        
        # Time NumPy implementation
        start_time = time.time()
        U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
        numpy_time = time.time() - start_time
        
        print(f"C++ time: {cpp_time:.4f}s, NumPy time: {numpy_time:.4f}s")
        print(f"Speedup: {numpy_time/cpp_time:.2f}x")
        print(f"Singular value difference: {abs(sigma - S_np[0]):.2e}")

if __name__ == "__main__":
    test_power_svd()
    benchmark_different_sizes()