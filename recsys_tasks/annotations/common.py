from dataclasses import dataclass
from typing import TypeVar, TypeAlias, NamedTuple, Generic

import numpy as np

SourceMatrix = TypeVar('SourceMatrix', bound=np.ndarray[[int, int], float])
ProductRepresentation = TypeVar('ProductRepresentation')

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
class ProductInfoWithDetails(ProductInfo):
    """Расширенное описание продукта."""

    similar_products_similarity: MatrixWithIndices | None = None
    used_products_preferences: MatrixWithIndices | None = None


@dataclass
class UserRecommendation(Generic[ProductRepresentation]):
    """Репрезентация пользовательских рекомендаций."""

    recommended_products: list[ProductRepresentation]
    not_recommended_products: list[ProductRepresentation] | None = None
