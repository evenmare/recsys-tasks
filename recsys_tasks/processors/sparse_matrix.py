"""Модуль, реализующий преобразования исходной матрицы в разраженные матрицы."""
from collections import Counter
from functools import cached_property
from typing import TypeVar, TypeAlias

import numpy as np

Shape: TypeAlias = tuple[int, int]
Size: TypeAlias = int

InputData = TypeVar('InputData', bound=np.ndarray[Shape, float])
LinearArray = TypeVar('LinearArray', bound=np.ndarray[Size, int])
Matrix = TypeVar('Matrix', bound=np.ndarray[Shape, int])


class SparseMatrixProcessor:
    """Процессор формирования разреженных матриц разных форматов."""

    source_data: InputData

    rows: LinearArray
    columns: LinearArray

    def __init__(self, source_data: InputData):
        """Конструктор класса."""
        self.source_data = source_data
        self.rows, self.columns = np.nonzero(self.source_data)

    @cached_property
    def separated(self) -> tuple[LinearArray, LinearArray, LinearArray]:
        """Вывод разраженной матрицы в формате: value, row, column.

        :return: value array, row array, column array
        """
        values = np.array(
            [
                self.source_data[x, y]
                for x, y in zip(tuple(self.rows), tuple(self.columns))
            ]
        )
        return values, self.rows, self.columns

    @cached_property
    def row_index(self) -> tuple[LinearArray, LinearArray, LinearArray]:
        """Вывод разреженной матрицы в формате: value, column, row_index.

        :return: value array, column array, row index array.
        """
        values, rows, columns = self.separated

        row_index: list[int] = [0 for _ in range(self.source_data.shape[0])]

        count = 1
        last_idx = rows[0]
        last_added_value = 0

        for idx in rows[1:]:
            if idx == last_idx:
                count += 1
                continue

            last_added_value += count
            row_index[idx] = last_added_value

            if idx - last_idx > 1:
                row_index[last_idx + 1:idx] = [last_added_value for _ in range(idx - last_idx - 1)]

            count = 1
            last_idx = idx

        return values, columns, np.array(row_index)

    @cached_property
    def value_columns(self) -> tuple[Matrix, Matrix]:
        """Вывод разреженной матрицы в формате матриц: value, column.

        :return: value matrix, column matrix.
        """
        values, rows, columns = self.separated

        [(_, matrices_cols)] = Counter(rows).most_common(1)
        values_mx, columns_mx = (
            np.zeros((self.source_data.shape[0], matrices_cols), dtype=float)
            for _ in range(2)
        )

        last_row = 0
        mx_col = 0
        for i, (row, column, value) in enumerate(zip(tuple(rows), tuple(columns), tuple(values))):
            if row != last_row:
                mx_col = 0
                last_row = row

            columns_mx[row, mx_col] = column
            values_mx[row, mx_col] = value

            mx_col += 1

        return values_mx, columns_mx

    def get_reformed_matrix(
        self,
        clear_zero_rated: bool = True,
        clear_inactive_users: bool = True,
        min_rating: float = 0,
    ) -> Matrix:
        """Генерация разреженной матрицы на основании правил.

        :param min_rating: Минимальный рейтинг продукта.
        :param clear_zero_rated: Очищать продукты с нулевым рейтингом?
        :param clear_inactive_users: Очищать "неактивных" пользователей?
        :return: Разреженная матрица предпочтений.
        """
        matrix = self.source_data

        if clear_zero_rated:
            matrix = matrix[~np.all(matrix == 0, axis=1)]

        if clear_inactive_users:
            matrix = matrix[:, ~np.all(matrix == 0, axis=0)]

        matrix = matrix[np.mean(matrix, axis=1) >= min_rating]

        return matrix
