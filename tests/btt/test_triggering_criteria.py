from bladesight.btt import triggering_criteria
import numpy as np
import pytest

@pytest.fixture
def arr_simple_ramp_up():
    """ Simplest ramp up signal.
        (n, n) -> (n, n) for n in [0, 10)
    Returns:
        np.ndarray: A simple ramp up signal.
        np.ndarray: The time array.
    """
    return np.arange(10), np.arange(10)

@pytest.fixture
def arr_simple_ramp_down():
    """ Simplest ramp down signal.
        (n, n) -> (n, n) for n in [0, 10)

    Returns:
        np.ndarray: A simple ramp up signal.
        np.ndarray: The time array.
    """
    return np.arange(10), np.array(list(np.arange(10))[::-1])


def test_threshold_crossing_interp(arr_simple_ramp_up):
    arr_t, arr_s = arr_simple_ramp_up
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 5)
    assert len(arr_toa) == 1
    assert arr_toa[0] == 5.0
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 4.9)
    assert len(arr_toa) == 1
    assert arr_toa[0] == 4.9
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 5.1)
    assert len(arr_toa) == 1
    assert arr_toa[0] == 5.1

def test_threshold_crossing_interp_simple_first_sample(arr_simple_ramp_up):
    arr_t, arr_s = arr_simple_ramp_up
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 0)
    assert len(arr_toa) == 0
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 0.1)
    assert len(arr_toa) == 1
    assert arr_toa[0] == 0.1
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 9)
    assert len(arr_toa) == 1
    assert arr_toa[0] == 9.0
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 9.1)
    assert len(arr_toa) == 0
    # Test that triggering on the falling edge does not work
    arr_toa = triggering_criteria.threshold_crossing_interp(
        arr_t, 
        arr_s, 
        5, 
        trigger_on_rising_edge=False
    )
    assert len(arr_toa) == 0

def test_threshold_crossing_interp_falling(arr_simple_ramp_down):
    arr_t, arr_s = arr_simple_ramp_down
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 5, trigger_on_rising_edge=False)
    assert len(arr_toa) == 1
    assert np.allclose(arr_toa[0], 4.0)
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 4.9, trigger_on_rising_edge=False)
    assert len(arr_toa) == 1
    assert np.allclose(arr_toa[0],4.1)
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 5.1, trigger_on_rising_edge=False)
    assert len(arr_toa) == 1
    assert np.allclose(arr_toa[0], 3.9)

def test_threshold_crossing_interp_simple_first_sample_falling(arr_simple_ramp_down):
    arr_t, arr_s = arr_simple_ramp_down
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 0, trigger_on_rising_edge=False)
    assert len(arr_toa) == 1
    assert np.allclose(arr_toa[0], 9.0)
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 0.1, trigger_on_rising_edge=False)
    assert len(arr_toa) == 1
    assert np.allclose(arr_toa[0], 8.9)
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 9, trigger_on_rising_edge=False)
    assert len(arr_toa) == 0
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 9.1, trigger_on_rising_edge=False)
    assert len(arr_toa) == 0
    # Test that triggering on the rising edge does not work
    arr_toa = triggering_criteria.threshold_crossing_interp(
        arr_t, 
        arr_s, 
        5, 
        trigger_on_rising_edge=True
    )
    assert len(arr_toa) == 0


# Test threshold_crossing_interp todo:
# - Test with n_est, ensure exception is raised if 
#    n_est is less than the number of ToAs
# - Test with falling edge triggering
# - Test with n_est and falling edge triggering
# - ...

    
