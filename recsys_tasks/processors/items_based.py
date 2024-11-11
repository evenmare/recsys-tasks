from functools import cached_property, lru_cache

import numpy as np

from recsys_tasks.annotations.common import (
    SourceMatrix,
    UserId,
    ProductId,
    MatrixWithIndices,
    UserRecommendation,
    ProductInfo, ProductInfoWithDetails,
)
from recsys_tasks.processors.cosine_similarity import CosineSimilarityProcessor


class ItemBasedRecommendationsProcessor:
    """Процессор генерации рекомендаций для пользователей на основе их опыта."""

    source_data: SourceMatrix
    user_id: UserId
    min_similarity_coefficient: float

    users_indices: list[UserId]
    products_indices: list[ProductId]

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

        self.users_indices = list(range(source_data.shape[1]))
        self.products_indices = list(range(source_data.shape[0]))
        self.mean_user_score = None

    @cached_property
    def not_rated_products(self) -> list[ProductId]:
        """Получение перечня продуктов, которые не оценил пользователь."""
        return np.argwhere(
            np.isnan(self.source_data[:, self.user_id])
        )[:, 0].tolist()

    @cached_property
    def similarity_matrix(self) -> MatrixWithIndices:
        """Построение матрицы подобия продуктов."""
        cosine_similarity_processor = CosineSimilarityProcessor(
            source_data=self.source_data,
            symmetric=True,
            default_value=np.nan,
        )
        similarity_matrix = cosine_similarity_processor()
        return MatrixWithIndices(
            matrix=similarity_matrix,
            columns=list(self.products_indices),
            index=list(self.products_indices),
        )

    @lru_cache
    def _get_not_similar_products(self, product_id: ProductId) -> list[ProductId]:
        """Получение непохожих продуктов."""
        product_similarities = self.similarity_matrix.matrix[product_id]
        return np.where(product_similarities < self.min_similarity_coefficient)[0].tolist()

    @lru_cache
    def _get_exclude_products(self, product_id: ProductId) -> list[ProductId]:
        """Получение продуктов для исключения из выборки (неоцененные + непохожие)."""
        not_rated_products = self.not_rated_products
        not_similar_products = self._get_not_similar_products(product_id)
        return list(set(not_similar_products) | set(not_rated_products))

    @lru_cache
    def _get_similar_rated_products(self, product_id: ProductId) -> list[ProductId]:
        """Получение оцененных похожих продуктов."""
        exclude_products = self._get_exclude_products(product_id)
        return [
            _product_idx for _product_idx in self.products_indices
            if _product_idx not in exclude_products
        ]

    def _get_similarity_of_similar_rated_products(self, product_id: ProductId) -> MatrixWithIndices:
        """Получение матрицы подобия для подобных продуктов."""
        exclude_products = self._get_exclude_products(product_id)

        similarities_for_selected_product = self.similarity_matrix.matrix[product_id]
        result_matrix = np.delete(similarities_for_selected_product, exclude_products)

        return MatrixWithIndices(
            matrix=result_matrix,
            columns=[product_id],
            index=self._get_similar_rated_products(product_id),
        )

    def _get_preferences_of_similar_products(self, product_id: ProductId) -> MatrixWithIndices:
        """Получение рейтингов похожих продуктов."""
        exclude_products = self._get_exclude_products(product_id)
        result_matrix = np.delete(self.source_data[:, self.user_id], exclude_products, axis=0)

        return MatrixWithIndices(
            matrix=result_matrix,
            columns=[self.user_id],
            index=self._get_similar_rated_products(product_id),
        )

    def __call__(self) -> UserRecommendation:
        """Расчет пользовательских рекомендаций."""
        self.is_new_user = not bool(np.any(~np.isnan(self.source_data[:, self.user_id])))

        if self.is_new_user:
            mean_products_score = np.nanmean(self.source_data, axis=1)
            top_product_index = np.argmax(mean_products_score)
            return UserRecommendation[ProductInfoWithDetails](
                recommended_products=[
                    ProductInfoWithDetails(
                        product_id=top_product_index,
                        calculated_score=mean_products_score[top_product_index],
                    )
                ],
            )

        self.mean_user_score = np.nanmean(self.source_data[:, self.user_id])

        recommended_products: list[ProductInfo] = []
        not_recommended_products: list[ProductInfo] = []

        for product_id in self.not_rated_products:
            products_similarity_info = self._get_similarity_of_similar_rated_products(product_id)
            products_preferences_info = self._get_preferences_of_similar_products(product_id)

            calculated_score = (
                np.sum(products_preferences_info.matrix * products_similarity_info.matrix) /
                np.sum(np.abs(products_similarity_info.matrix))
            )

            product_details = ProductInfoWithDetails(
                product_id=product_id,
                calculated_score=calculated_score,
                similar_products_similarity=products_similarity_info,
                used_products_preferences=products_preferences_info,
            )

            if calculated_score > self.mean_user_score:
                recommended_products.append(product_details)
            else:
                not_recommended_products.append(product_details)

        return UserRecommendation[ProductInfoWithDetails](
            recommended_products=recommended_products,
            not_recommended_products=not_recommended_products,
        )
