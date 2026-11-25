import pytest
from sample_code.average import calculate_average


def test_average_of_three_numbers():
    # Arrange
    numbers = [2, 4, 6]
    # Act
    actual = calculate_average(numbers)
    # Assert
    assert actual == 4


def test_average_of_one_number():
    assert calculate_average([5]) == 5


def test_average_of_empty_list():
    with pytest.raises(ValueError):
        calculate_average([])
