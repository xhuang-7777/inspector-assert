import pytest
from sample_code.calculator import Calculator, CalculationError


def test_addition():
    # Arrange: choose the input values and the expected result.
    calculator = Calculator()
    # Act: call the operation.
    actual = calculator.add(2, 3)
    # Assert: compare the actual result with the requirement.
    assert actual == 5


def test_negative_subtraction():
    calculator = Calculator()
    assert calculator.subtract(2, 5) == -3


def test_negative_multiplication():
    calculator = Calculator()
    assert calculator.multiply(-3, 4) == -12


def test_whole_number_division():
    calculator = Calculator()
    assert calculator.divide(8, 2) == 4


def test_fractional_division():
    calculator = Calculator()
    assert calculator.divide(7, 2) == 3.5


def test_division_by_zero():
    calculator = Calculator()
    with pytest.raises(CalculationError):
        calculator.divide(5, 0)


def test_valid_operation_after_error():
    calculator = Calculator()
    with pytest.raises(CalculationError):
        calculator.divide(5, 0)
    assert calculator.add(2, 3) == 5
