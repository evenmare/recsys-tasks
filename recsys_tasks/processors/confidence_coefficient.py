from typing import Self

__all__ = (
    'MD',
    'MND',
    'E',
)

class BaseCoefficient:
    """Базовая логика для коэффициентов уверенности / неуверенности."""

    _value: float

    def __init__(self, coefficient_value: float):
        """Конструктор класса."""
        self._value = coefficient_value

    def __float__(self):
        """Репрезентация значения в float."""
        return self._value

    def __and__(self, other: 'BaseCoefficient') -> Self:
        """Реализация правила нечеткой логики: И."""
        return self.__class__(min(float(self), float(other)))

    def __or__(self, other: 'BaseCoefficient') -> Self:
        """Реализация правила нечеткой логики: ИЛИ."""
        return self.__class__(max(float(self), float(other)))

    def __invert__(self):
        """Реализация правила нечеткой логики: НЕ."""
        return self.__class__(1 - float(self))

    def __add__(self, other: 'BaseCoefficient') -> Self:
        """Реализация сложения коэффициентов."""
        return self.__class__(float(self) + float(other))

    def __sub__(self, other: 'BaseCoefficient') -> Self:
        """Реализация вычитания коэффициентов."""
        return self.__class__(float(self) - float(other))

    def __mul__(self, other: 'BaseCoefficient') -> Self:
        """Реализация умножения коэффициентов."""
        return self.__class__(float(self) * float(other))

    def __repr__(self):
        """Репрезентация значения коэффициента."""
        return f'{self.__class__.__name__}={float(self)}'


class MD(BaseCoefficient):
    """Мера доверия."""



class MND(BaseCoefficient):
    """Мера недоверия."""


class E:
    """Предположение."""

    def __init__(self, md: MD, mnd: MND):
        """Конструктор класса."""
        self._md = md
        self._mnd = mnd

    @property
    def md(self):
        """Мера доверия."""
        return self._md

    @property
    def mnd(self):
        """Мера недоверия."""
        return self._mnd

    def __repr__(self):
        """Репрезентация."""
        return f'{self.__class__.__name__}: {self.md}, {self.mnd}'
