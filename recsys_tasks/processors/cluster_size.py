"""Модуль, реализующий поэтапное объединение по заданному размеру кластера."""
from typing import TypeVar, TypeAlias

import numpy as np
from attr import dataclass

from recsys_tasks.processors.cosine_similarity import CosineSimilarityProcessor

Shape = tuple[int, int]

InputData = TypeVar('InputData', bound=np.ndarray[Shape, float])
UnionMatrix = TypeVar('UnionMatrix', bound=np.ndarray[Shape, float])
UnionIndexDetails: TypeAlias = tuple[list[int], list[int]]


@dataclass
class UnionMatrixStage:
    """Структура, описывающая этап объединения."""

    index_map: dict[int, list[int]]
    union_indices: UnionIndexDetails
    result_matrix: UnionMatrix
    value: float


class ClusterSizeProcessor:
    """Процессор метода ближайших соседей по размеру кластера."""

    source_data: InputData
    union_history: list[UnionMatrixStage]

    def __init__(self, source_data: InputData):
        """Конструктор класса."""
        self.source_data = source_data
        self.union_history = []

    def __call__(self, cluster_size: float) -> list[UnionMatrixStage]:
        """Формирование объединений из размера кластера.

        :param cluster_size: Размер кластера.
        :return: Этапы объединений.
        """
        last_matrix = self.source_data

        absolute_indices_map: dict[int, list[int]] = {i: [i] for i in range(last_matrix.shape[0])}

        for i in range(self.source_data.shape[0]):
            union_idx_pair, current_max_value = CosineSimilarityProcessor.get_top_values(last_matrix)[0]
            if current_max_value < cluster_size:
                break

            new_union_matrix_shape = (last_matrix.shape[0] - 1, last_matrix.shape[1] - 1)
            new_union_matrix = np.full(new_union_matrix_shape, -1, dtype=float)

            indices_pairs = (
                (core_index, slave_index)
                for core_index in range(last_matrix.shape[0])
                for slave_index in range(core_index + 1, last_matrix.shape[0])
                if (core_index, slave_index) != union_idx_pair
            )
            union_indices_set = frozenset(union_idx_pair)

            new_union_index = min(union_idx_pair)
            indices_changes_from = max(union_idx_pair)

            union_values = {}
            for core_index, slave_index in indices_pairs:
                unique_index = None

                old_index_new_index_map = {
                    core_index: core_index,
                    slave_index: slave_index,
                }
                if core_index > indices_changes_from:
                    old_index_new_index_map[core_index] -= 1
                if slave_index > indices_changes_from:
                    old_index_new_index_map[slave_index] -= 1

                iteration_coord_set = {core_index, slave_index}
                if set(union_indices_set) & iteration_coord_set:
                    old_unique_index = next(iter(iteration_coord_set - union_indices_set))
                    unique_index = old_index_new_index_map[old_unique_index]

                x, y = old_index_new_index_map[core_index], old_index_new_index_map[slave_index]
                if unique_index is None:
                    new_union_matrix[x, y] = last_matrix[core_index, slave_index]
                elif (existing_value := union_values.get(unique_index)) is not None:
                    union_values[unique_index] = max(existing_value, float(last_matrix[core_index, slave_index]))
                else:
                    union_values[unique_index] = float(last_matrix[core_index, slave_index])

            for new_slave_index, value in union_values.items():
                if new_union_index > new_slave_index:
                    new_union_matrix[new_slave_index, new_union_index] = value
                else:
                    new_union_matrix[new_union_index, new_slave_index] = value

            new_indices_map = {
                j: (
                    [*absolute_indices_map[j], *absolute_indices_map[union_idx_pair[1]]]
                    if j == new_union_index
                    else (
                        absolute_indices_map[j]
                        if j < indices_changes_from
                        else absolute_indices_map[j + 1]
                    )
                )
                for j in range(new_union_matrix_shape[0])
            }

            self.union_history.append(
                UnionMatrixStage(
                    index_map=new_indices_map,
                    union_indices=(absolute_indices_map[union_idx_pair[0]], absolute_indices_map[union_idx_pair[1]]),
                    result_matrix=new_union_matrix,
                    value=current_max_value,
                )
            )

            last_matrix = new_union_matrix
            absolute_indices_map = new_indices_map

        return self.union_history

    @property
    def dendrogram_info(self):
        """Построение дендрограммы."""
        formatted_history = []

        last_iteration = self.union_history[-1]
        next_figured_index = next(
            (
                last_iteration.index_map[indices_key][0]
                for indices_key in last_iteration.index_map
                if len(last_iteration.index_map[indices_key]) == 1
            ),
            self.source_data.shape[0],
        )
        indices_tuples_figured_index: dict[tuple[int, ...], int] = {}

        for stage in self.union_history:
            union_info = []

            flatten_union_index = []
            for union_indices in stage.union_indices:
                if len(union_indices) == 1:
                    union_info.append(union_indices[0])
                else:
                    union_info.append(indices_tuples_figured_index[tuple(union_indices)])

                flatten_union_index.extend(union_indices)

            cluster_size = len(flatten_union_index)
            indices_tuples_figured_index[tuple(flatten_union_index)] = next_figured_index

            union_info.extend([stage.value, cluster_size])
            formatted_history.append(union_info)

            next_figured_index += 1

        return np.array(formatted_history)
