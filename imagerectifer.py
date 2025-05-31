import numpy as np


class ImageRectifier:
    """
    Класс для выполнения аффинных преобразований изображений.

    Аффинное преобразование описывается матрицей 'a' и вектором смещения 'offset'.
    Поддерживает прямое и обратное преобразование координат.

    Attributes:
        a: Матрица аффинного преобразования (2x2)
        offset: Вектор смещения (2,)
        a_inv (np.ndarray): Обратная матрица преобразования
        offset_inv (np.ndarray): Обратный вектор смещения
    """

    def __init__(self, a, offset):
        """
        Инициализирует аффинное преобразование.

        Args:
            a: Матрица преобразования 2x2
            offset: Вектор смещения [dx, dy]
        """
        self.a = a
        self.offset = offset
        self.a_inv = np.linalg.inv(a)
        self.offset_inv = self.a_inv.dot(self.offset)

    def apply_transform(self, src: np.ndarray, height: float, width: float) -> np.ndarray:
        """
        Применяет аффинное преобразование к изображению.

        Args:
            src: Исходное изображение (2D массив)
            height: Высота результирующего изображения
            width: Ширина результирующего изображения

        Returns:
            Преобразованное изображение
        """
        rows, cols = src.shape
        # Создаем координатную сетку
        points = np.mgrid[0:np.uint16(cols), 0:np.uint16(rows)].reshape((2, rows*cols))

        # Применяем преобразование
        transformed_points = self.a.dot(points).round().astype('int32')
        transformed_points[0] += self.offset[0]
        transformed_points[1] += self.offset[1]

        # Преобразуем координаты и создаем новое изображение
        x_coords, y_coords = transformed_points.reshape((2, rows, cols), order='F')
        indices = x_coords + cols * y_coords
        return np.take(src, indices, mode='wrap')[:int(height) + 1, :int(width) + 1]

    def inverse_transform(self, x: int, y: int) -> np.ndarray:
        """
        Выполняет обратное аффинное преобразование координат.

        Args:
            x: X-координата точки
            y: Y-координата точки

        Returns:
            Массив [x', y'] - преобразованные координаты
        """
        transformed_point = self.a_inv.dot([x, y])
        transformed_point -= self.offset_inv
        return transformed_point.round().astype('int32')
