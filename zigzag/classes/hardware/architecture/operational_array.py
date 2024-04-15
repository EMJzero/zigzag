import numpy as np
from typeguard import typechecked
from zigzag.classes.hardware.architecture.Dimension import Dimension
from zigzag.classes.hardware.architecture.operational_unit import (
    OperationalUnit,
    Multiplier,
)


@typechecked
class OperationalArray:
    """!  This class captures multi-dimensional operational array size."""

    def __init__(self, operational_unit: OperationalUnit, dimensions: dict[str, int]):
        """!  The class constructor
        @param operational_unit: an OperationalUnit object including precision and single operation energy, later we
        can add idle energy also (e.g. for situations that one or two of the input operands is zero).
        @param dimensions: define the name and size of each multiplier array dimensions, e.g. {'D1': 3, 'D2': 5}.
        """
        self.unit = operational_unit
        self.total_unit_count = int(np.prod(list(dimensions.values())))
        try:
            self.total_area = operational_unit.area * self.total_unit_count
        except TypeError:  # branch for IMC
            self.total_area = operational_unit.area

        self.dimensions: list[Dimension] = [
            Dimension(idx, name, size) for idx, (name, size) in enumerate(dimensions.items())
        ]
        # self.dimension_sizes = [dim.size for dim in base_dims]
        # self.nb_dimensions = len(base_dims)

    # JSON Representation of this class to save it to a json file.
    def __jsonrepr__(self):
        return {"operational_unit": self.unit, "dimensions": self.dimensions}

    def __eq__(self, other) -> bool:
        if not isinstance(other, OperationalArray):
            return False
        return self.unit == other.unit and self.dimensions == other.dimensions


class MultiplierArray(OperationalArray):
    """!  Description missing"""

    def __init__(
        self,
        multiplier: Multiplier,
        dimensions: dict[str, int],
        operand_spatial_sharing: dict[str, set[tuple[int, ...]]] = {},
    ):
        super(MultiplierArray, self).__init__(multiplier, dimensions)
        self.multiplier = self.unit
        self.operand_spatial_sharing = operand_spatial_sharing


def multiplier_array_example1():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {"D1": 14, "D2": 3, "D3": 4}
    operand_spatial_sharing = {
        "I1": {(1, 0, 0)},
        "O": {(0, 1, 0)},
        "I2": {(0, 0, 1), (1, 1, 0)},
    }
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    return multiplier_array


def multiplier_array_example2():
    """Multiplier array variables"""
    multiplier_input_precision = [8, 8]
    multiplier_energy = 0.5
    multiplier_area = 0.1
    dimensions = {"D1": 14, "D2": 12}
    operand_spatial_sharing = {"I1": {(1, 0)}, "O": {(0, 1)}, "I2": {(1, 1)}}
    multiplier = Multiplier(multiplier_input_precision, multiplier_energy, multiplier_area)
    multiplier_array = MultiplierArray(multiplier, dimensions, operand_spatial_sharing)

    return multiplier_array
