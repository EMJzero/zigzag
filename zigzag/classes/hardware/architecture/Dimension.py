import re

from typeguard import typechecked


@typechecked
class Dimension:
    """! Operational Array Dimension"""

    def __init__(self, index: int = 0, name: str = "", size: int = 1):
        """!  The class constructor
        @param index: The integer index of this Dimension.
        @param name: The user-provided name of this Dimension.
        @param size: The user-provided size of this Dimension.
        """
        self.id = index
        self.name = name
        self.size = size

    def __str__(self):
        return self.name
        # return f"Dimension(id={self.id},name={self.name},size={self.size})"

    def __repr__(self):
        return str(self)

    def __jsonrepr__(self):
        """!  JSON representation of this class to save it to a json file."""
        return self.__dict__

    def __eq__(self, other):
        # id should be enough to identify dimension
        if not isinstance(other, Dimension):
            return False
        return self.id == other.id
        # return other.id == self.id and self.name == other.name and self.size == other.size

    def __hash__(self):
        # id should be enough to identify dimension
        return hash(self.id)
        # return hash(self.id) ^ hash(self.name)

    @staticmethod
    def parse_user_input(x: str):
        assert bool(re.match(r"D\d", x)), f"User specified dimension {x} not recognized"
        # Dimension id starts at 0, user format starts at D1
        idx: int = int(re.findall(r"\d", x).pop()) - 1
        return Dimension(idx, x)
