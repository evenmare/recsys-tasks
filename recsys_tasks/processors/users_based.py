from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, TypeAlias, NamedTuple

import numpy as np

from recsys_tasks.processors.cosine_similarity import CosineSimilarityProcessor

SourceMatrix = TypeVar('SourceMatrix', bound=np.ndarray[[int, int], float])

UserId: TypeAlias = int
ProductId: TypeAlias = int


class MatrixWithIndices(NamedTuple):
    """Описание матрицы."""

    matrix: np.ndarray[[int, int], float]
    columns: list[int]
    index: list[int]


@dataclass
class ProductInfo:
    """Описание продукта."""

    product_id: ProductId
    calculated_score: float


@dataclass
class UserRecommendation:
    """Репрезентация пользовательских рекомендаций."""

    recommended_products: list[ProductInfo]
    not_recommended_products: list[ProductInfo] | None = None


class UserBasedRecommendationsProcessor:
    """Процессор генерации рекомендаций для пользователей на основе их опыта."""

    source_data: SourceMatrix
    user_id: UserId
    min_similarity_coefficient: float

    users_indices: tuple[UserId]
    products_indices: tuple[ProductId]

    def __init__(
        self,
        source_data: SourceMatrix,
        user_id: UserId,
        *,
        min_similarity_coefficient: float = 0.8,
    ):
        """Конструктор класса."""
        self.source_data = source_data
        self.user_id = user_id
        self.min_similarity_coefficient = min_similarity_coefficient

        self.users_indices = tuple(range(source_data.shape[1]))
        self.products_indices = tuple(range(source_data.shape[0]))
        self.mean_user_score = None

    @cached_property
    def not_rated_products(self) -> list[ProductId]:
        """Получение перечня продуктов, которые не оценил пользователь."""
        return np.argwhere(
            np.isnan(self.source_data[:, self.user_id])
        )[:, 0].tolist()

    @cached_property
    def all_users_not_rated_needed_products(self) -> list[UserId]:
        """Получение группы пользователей, не оценивших те же продукты, что и пользователь (включая его самого)."""
        return np.argwhere(
            np.any(
                np.isnan(self.source_data[self.not_rated_products, :]),
                axis=0
            )
        )[:, 0].tolist()

    @cached_property
    def users_not_rated_needed_products(self) -> list[UserId]:
        """Получение перечня пользователей, не оценившие те же продукты, что и рассматриваемый пользователь."""
        return [
            user_idx for user_idx in self.all_users_not_rated_needed_products
            if user_idx != self.user_id
        ]

    @cached_property
    def user_preferences_matrix_for_similarity(self) -> MatrixWithIndices:
        """Построение матрицы предпочтений для дальнейшего расчета косинусного подобия.

        Исключаются: пользователи, которые не оценили продукт <product_id> за исключением пользователя <user_id>.
        """
        exclude_columns = self.users_not_rated_needed_products
        result_matrix = np.delete(self.source_data, exclude_columns, axis=1)
        result_matrix[np.isnan(result_matrix)] = 0
        return MatrixWithIndices(
            matrix=result_matrix,
            columns=[user_id for user_id in self.users_indices if user_id not in exclude_columns],
            index=list(self.products_indices),
        )

    @cached_property
    def rated_user_preferences_matrix(self) -> MatrixWithIndices:
        """Построение матрицы предпочтений пользователей, которые оценили определенный продукт.

        Исключаются: пользователи, которые не оценили продукт <product_id>.
        """
        exclude_columns = self.all_users_not_rated_needed_products
        result_matrix = np.delete(self.source_data, exclude_columns, axis=1)
        return MatrixWithIndices(
            matrix=result_matrix,
            columns=[user_id for user_id in self.users_indices if user_id not in exclude_columns],
            index=list(self.products_indices),
        )

    @cached_property
    def similarity_matrix_of_users_rated(self) -> MatrixWithIndices:
        """Построение матрицы подобия пользователей."""
        preferences_data = self.user_preferences_matrix_for_similarity

        # Не учитываем оценки по продуктам, которые не были оценены пользователем.
        preferences_matrix = preferences_data.matrix.copy()
        preferences_matrix = np.delete(preferences_matrix, self.not_rated_products, axis=0)

        cosine_similarity_processor = CosineSimilarityProcessor(
            source_data=preferences_matrix.T,
            symmetric=True,
            default_value=np.nan,
        )
        similarity_matrix = cosine_similarity_processor()

        indices = preferences_data.columns.copy()
        return MatrixWithIndices(
            matrix=similarity_matrix,
            columns=indices,
            index=indices,
        )

    @cached_property
    def different_users(self) -> list[UserId]:
        """Получение непохожих пользователей."""
        similarity_matrix_data = self.similarity_matrix_of_users_rated

        matrix_user_index = similarity_matrix_data.columns.index(self.user_id)

        different_users = np.argwhere(
            similarity_matrix_data.matrix[matrix_user_index, :] < self.min_similarity_coefficient
        )
        return different_users[~np.all(different_users == matrix_user_index, axis=1), :][:, 0].tolist()


    @cached_property
    def similar_users_similarity_matrix(self) -> MatrixWithIndices:
        """Построение матрицы подобия пользователей для похожих пользователей.

        Исключаются: пользователи, подобие с которыми < <self.min_similarity_coefficient>.
        """
        different_users = self.different_users

        similarity_matrix_data = self.similarity_matrix_of_users_rated
        result_matrix = np.delete(
            similarity_matrix_data.matrix,
            different_users,
            axis=0,
        )
        result_matrix = np.delete(
            result_matrix,
            different_users,
            axis=1,
        )

        source_indices = self.user_preferences_matrix_for_similarity.columns
        new_similarity_matrix_indices = [
            idx for i, idx in enumerate(source_indices)
            if i not in different_users
        ]

        return MatrixWithIndices(
            matrix=result_matrix,
            columns=new_similarity_matrix_indices,
            index=new_similarity_matrix_indices,
        )

    @cached_property
    def similar_rated_users_preferences_matrix(self) -> MatrixWithIndices:
        """Построение матрицы оценок товаров пользователей, похожих на анализируемого пользователя.

        Исключаются: пользователи, подобие с которыми < <self.min_similarity_coefficient>.
        """
        rated_users_preferences_data = self.rated_user_preferences_matrix

        users_rated = rated_users_preferences_data.columns
        similar_users = self.similar_users_similarity_matrix.columns

        different_users_indices = [
            i for i, idx in enumerate(users_rated)
            if idx not in similar_users
        ]

        result_matrix = np.delete(rated_users_preferences_data.matrix, different_users_indices, axis=1)
        return MatrixWithIndices(
            matrix=result_matrix,
            columns=[i for i in similar_users if i != self.user_id],
            index=rated_users_preferences_data.index,
        )

    def __call__(self):
        """Расчет пользовательских рекомендаций."""
        self.is_new_user = not bool(np.any(~np.isnan(self.source_data[:, self.user_id])))

        if self.is_new_user:
            mean_products_score = np.nanmean(self.source_data, axis=1)
            top_product_index = np.argmax(mean_products_score)
            return UserRecommendation(
                recommended_products=[
                    ProductInfo(
                        product_id=top_product_index,
                        calculated_score=mean_products_score[top_product_index],
                    )
                ],
            )

        self.mean_user_score = np.nanmean(self.source_data[:, self.user_id])

        recommended_products: list[ProductInfo] = []
        not_recommended_products: list[ProductInfo] = []

        rated_users_preferences_matrix = self.rated_user_preferences_matrix.matrix
        similar_users_similarity_matrix_info = self.similar_users_similarity_matrix
        similar_rated_users_preferences_matrix = self.similar_rated_users_preferences_matrix.matrix

        similarity_matrix, similarity_matrix_user_id = (
            similar_users_similarity_matrix_info.matrix,
            similar_users_similarity_matrix_info.columns.index(self.user_id),
        )

        for product_id in self.not_rated_products:
            mean_product_score = np.mean(rated_users_preferences_matrix)
            similarity = similarity_matrix[
                ~np.isnan(similarity_matrix[:, similarity_matrix_user_id]),
                similarity_matrix_user_id
            ]
            score_difference = similar_rated_users_preferences_matrix[product_id, :] - mean_product_score

            calculated_score = (
                mean_product_score + np.sum(score_difference * similarity) /
                np.sum(np.abs(similarity))
            )

            if calculated_score > self.mean_user_score:
                recommended_products.append(
                    ProductInfo(
                        product_id=product_id,
                        calculated_score=calculated_score,
                    )
                )
            else:
                not_recommended_products.append(
                    ProductInfo(
                        product_id=product_id,
                        calculated_score=calculated_score,
                    )
                )

        return UserRecommendation(
            recommended_products=recommended_products,
            not_recommended_products=not_recommended_products,
        )
