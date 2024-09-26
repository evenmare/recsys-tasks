"""Модуль, реализующий логику по работе с косинусным подобием."""

from typing import TypeVar

import numpy as np

Shape = tuple[int, int]

InputData = TypeVar('InputData', bound=np.ndarray[Shape, int])
CalculatedSimilarity = TypeVar('CalculatedSimilarity', bound=np.ndarray[Shape, float])

ElementId = TypeVar('ElementId', bound=tuple[int, int])
Score = TypeVar('Score', bound=float)


class CosineSimilarityProcessor:
    """Класс, реализующий логику по подсчету косинусного подобия.

    Реализуется:
        - Логика по заполнению NxN матрицы по показателю косинусного подобия.
        - Получение похожих пересечений.
    """

    source_data: InputData

    @classmethod
    def get_top_values(
        cls,
        calculated_similarity: CalculatedSimilarity,
        count: int = 1,
    ) -> list[tuple[ElementId, Score]]:
        """Получение топовых значений из матрицы подобия.

        :param calculated_similarity: Матрица подобия.
        :param count: Число необходимых похожих элементов.
        :return: Список объектов, представляющих похожие значения.
        """
        result = []

        members_count: int = calculated_similarity.shape[0]
        arrayed_form_length: int = members_count ** 2
        arrayed_form: np.ndarray[int, float] = calculated_similarity.reshape(arrayed_form_length)

        sorted_indices_of_arrayed_form: np.ndarray = np.argsort(arrayed_form)
        for arrayed_form_index in sorted_indices_of_arrayed_form[::-1][:count]:
            x, y = divmod(int(arrayed_form_index), members_count)
            result.append(((x, y), float(calculated_similarity[x, y])))

        return result

    def __init__(
        self,
        source_data: InputData,
    ):
        """Конструктор класса."""
        self.source_data = source_data

    def __call__(self) -> CalculatedSimilarity:
        """Подсчет косинусного подобия для набора данных."""
        members_count = self.source_data.shape[0]

        similarity_matrix_shape: tuple[int, int] = (members_count, members_count)
        default_value: float = -1.  # -1 <= cos <= 1

        similarity_matrix: CalculatedSimilarity = np.full(similarity_matrix_shape, default_value)

        indices_pairs = (
            (core_index, slave_index)
            for core_index in range(members_count)
            for slave_index in range(core_index + 1, members_count)
        )
        for core_index, slave_index in indices_pairs:
            core_line, slave_line = self.source_data[core_index], self.source_data[slave_index]
            similarity_value = sum(core_line * slave_line) / (sum(core_line ** 2) ** 0.5 * sum(slave_line ** 2) ** 0.5)
            similarity_matrix[core_index, slave_index] = similarity_value

        return similarity_matrix
