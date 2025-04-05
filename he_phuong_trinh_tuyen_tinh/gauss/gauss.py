import numpy as np
from tabulate import tabulate


# Hàm giải hệ phương trình tuyến tính bằng phương pháp khử Gauss
def solve_gauss(A, b):
    n = len(A)
    # Tạo ma trận mở rộng
    augmented_matrix = np.concatenate((A, b.reshape(n, 1)), axis=1)

    # Giai đoạn loại bỏ
    for i in range(n-1):
        # Chọn pivot
        pivot_row = i
        for j in range(i+1, n):
            if abs(augmented_matrix[j, i]) > abs(augmented_matrix[pivot_row, i]):
                pivot_row = j

        # Hoán đổi hàng pivot
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        # Loại bỏ các ẩn số bên dưới pivot
        for j in range(i+1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j] -= factor * augmented_matrix[i]

        # In ma trận mở rộng sau từng bước biến đổi
        print(f"🔹 Ma trận mở rộng sau bước biến đổi {i+1}:")
        print(tabulate(augmented_matrix, tablefmt="grid", floatfmt=".6f"))

    # Giai đoạn quay lui
    x = np.zeros(n)
    x[n-1] = augmented_matrix[n-1, n] / augmented_matrix[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (augmented_matrix[i, n] - np.dot(augmented_matrix[i, i+1:n], x[i+1:n])) / augmented_matrix[i, i]

    return augmented_matrix, x

# Đọc ma trận mở rộng từ file
def read_augmented_matrix_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        rows = len(lines)
        cols = len(lines[0].split()) - 1  # Trừ 1 cột cuối cùng là cột tự do
        matrix = np.zeros((rows, cols))
        b = np.zeros(rows)
        for i in range(rows):
            line_values = [float(x) for x in lines[i].split()]
            matrix[i] = line_values[:-1]
            b[i] = line_values[-1]
        return matrix, b

# Đường dẫn tới file chứa ma trận mở rộng
filename = "augmented_matrix.txt"

# Đọc ma trận mở rộng từ file
A, b = read_augmented_matrix_from_file(filename)

# Giải hệ phương trình bằng phương pháp khử Gauss
augmented_matrix, x = solve_gauss(A, b)

# Tính rank của ma trận hệ số và ma trận mở rộng bậc thang
rank_A = np.linalg.matrix_rank(A)
rank_augmented_matrix = np.linalg.matrix_rank(augmented_matrix[:, :-1])
# In rank của ma trận hệ số và ma trận mở rộng
print("🔹 Rank của ma trận hệ số A:", rank_A)
print("🔹 Rank của ma trận mở rộng:", rank_augmented_matrix)
# In ma trận mở rộng sau từng bước biến đổi
print("🔹 Ma trận mở rộng sau phương pháp khử Gauss:")
print(tabulate(augmented_matrix, tablefmt="grid", floatfmt=".6f"))

# Kiểm tra nghiệm
if rank_A == rank_augmented_matrix:
    if rank_A == len(A[0]):
        print("🔹 Hệ phương trình có duy nhất một nghiệm:")
        print(tabulate([[f"x{i+1}", x[i]] for i in range(len(x))], headers=["Biến", "Giá trị"], tablefmt="grid", floatfmt=".6f"))
    else:
        print("🔹 Hệ phương trình có vô số nghiệm")
else:
    print("🔹 Hệ phương trình vô nghiệm")
