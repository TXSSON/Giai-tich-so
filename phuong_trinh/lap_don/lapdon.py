import numpy as np
from tabulate import tabulate
from math import sin, pi

def fixed_point_iteration(phi, a, b, x0, epsilon, N0):
    # Bước 1: Kiểm tra khoảng cách ly nghiệm
    def derivative_max(phi, a, b, num_points=1000):
        x_vals = np.linspace(a, b, num_points)
        deriv_vals = np.abs(np.gradient([phi(x) for x in x_vals], x_vals))
        return np.max(deriv_vals)
    
    q = derivative_max(phi, a, b)
    
    # Bước 3: Nếu q >= 1 thì dừng
    if q >= 1:
        print("Điều kiện hội tụ không đảm bảo, dừng chương trình.")
        return None
    
    # Bước 4: Khởi tạo
    i = 1
    epsilon = (epsilon * (1 - q)) / q
    iterations = [[0, x0, ""]]
    
    # Bước 5: Lặp
    while i <= N0:
        x = phi(x0)  # Bước 5.1
        diff = abs(x - x0)
        # Lưu giá trị với 6 chữ số sau dấu thập phân
        iterations.append([i, x, diff])
        
        if diff < epsilon:  # Bước 5.2
            print(tabulate(iterations, headers=["k", "x_k", "|x_k+1 - x_k|"], tablefmt="grid",floatfmt=".9f"))
            return x
        x0 = x  # Bước 5.3
        i += 1
    
    # Bước 6: Dừng do số lần lặp tối đa
    print("Số lần lặp đạt tối đa, dừng chương trình.")
    print(tabulate(iterations, headers=["k", "x_k", "|x_k+1 - x_k|"], tablefmt="grid",floatfmt=".9f"))
    return x

# Ví dụ sử dụng
def phi_example(x):
    return sin(x) +  pi/4

a = 1.5
b =2
x0 = 2
eps = 0.027619
N0 = 100
result = fixed_point_iteration(phi_example, a, b, x0,eps, N0)
if result is not None:
    print("Nghiệm gần đúng:", result)
