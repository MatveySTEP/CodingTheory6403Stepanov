import numpy as np


###1.1. Реализовать функцию REF(), приводящую матрицу к ступенчатому виду.
def REF(matrix):
    rows, cols = matrix.shape
    lead = 0

    for r in range(rows):
        if lead >= cols:
            break

        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    return matrix[:r]

        matrix[[i, r]] = matrix[[r, i]]

        for i in range(r + 1, rows):
            if matrix[i, lead] == 1:
                matrix[i] ^= matrix[r]

        lead += 1

    return matrix[:r]


###1.2. Реализовать функцию RREF(), приводящую матрицу к приведённому ступенчатому виду.
def RREF(matrix):
    rows, columns = matrix.shape
    pivot_column = 0

    for row_index in range(rows):
        if pivot_column >= columns:
            return matrix

        # Find the first non-zero element in current column
        swap_row = row_index
        while matrix[swap_row, pivot_column] == 0:
            swap_row += 1
            if swap_row == rows:
                swap_row = row_index
                pivot_column += 1
                if pivot_column == columns:
                    return matrix

        # Swap rows
        matrix[[swap_row, row_index]] = matrix[[row_index, swap_row]]

        # Normalize leading element
        if matrix[row_index, pivot_column] != 0:
            matrix[row_index] = matrix[row_index] / matrix[row_index, pivot_column]

        # Clear elements in this column
        for row in range(rows):
            if row != row_index and matrix[row, pivot_column] == 1:
                matrix[row] = np.logical_xor(matrix[row], matrix[row_index]).astype(int)

        pivot_column += 1

    return matrix


# 1.3.1 На основе входной матрицы сформировать порождающую матрицу в ступенчатом виде.
def Linear():
    S = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                  [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                  [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
    S_REF = RREF(S)
    print(S_REF, "\n")

    ###1.3.2 Задать n равное числу столбцов и k равное числу строк полученной матрицы (без учёта полностью нулевых строк).
    mask = np.any(S_REF, axis=1)
    filtered_matrix = S_REF[mask]

    dimensions = filtered_matrix.shape
    print(f"Rows after filtering: {dimensions[1]}")
    print(f"Non-zero rows count: {dimensions[0]}")
    # 1.3.3 Сформировать проверочную матрицу на основе порождающей.
    # Шаг 1. Сформировать матрицу 𝐆∗ в приведённом ступенчатом виде на основе порождающей.
    G = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    G_REF = RREF(G)
    print(G_REF)
    # Шаг 2. Зафиксировать ведущие столбцы lead матрицы 𝐆∗.
    rows, cols = G_REF.shape
    pivot_cols = []
    current_row = 0
    current_col = 0

    while current_row < rows and current_col < cols:
        # Find the first non-zero element in current row
        while current_col < cols and G_REF[current_row, current_col] == 0:
            current_col += 1

        # If leading element found, add column index to list
        if current_col < cols:
            pivot_cols.append(current_col)

        # Move to next row and column
        current_row += 1
        current_col += 1

    print(f"\nNumbers lead cols: {pivot_cols}")

    # Шаг 3. Сформировать сокращённую матрицу 𝐗, удалив ведущие столбцы матрицы 𝐆∗.
    keep_indices = list(set(range(G_REF.shape[1])) - set(pivot_cols))
    X = G_REF[:, keep_indices]
    print("\nAbbreviated matrix:")
    print(X)
    #Create matrix I
    I = np.eye(len(X[0]), dtype=int)
    # Шаг 4. Сформировать матрицу 𝐇, поместив в строки, соответствующие позициям ведущих столбцов строки из 𝐗, а в остальные – строки единичной матрицы.
    H = np.zeros((len(X) + len(I), len(X[0])), dtype=int)
    j = 0
    r = 0
    for i in range(len(X) + len(I)):
        if i in pivot_cols:
            H[i, :] = X[r, :]
            r += 1
        else:
            H[i, :] = I[j, :]
            j += 1
    print("\nMatrix H:\n", H)



def main():
    # Example usage
    matrix = np.array([[0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 0],
                       [0, 1, 0, 1, 1],
                       [0, 1, 0, 1, 0]], dtype=int)

    result1 = REF(matrix)
    result2 = RREF(matrix)
    print("result1:\n", result1, "\nresult2:\n", result2, "\n")
    Linear()



if __name__ == '__main__':
    main()
