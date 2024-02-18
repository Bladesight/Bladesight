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

@pytest.fixture
def arr_zig_zag():
    """ A zig zag signal that is supposed to trigger multiple times
    when using the simple threshold.

    Returns:
        np.ndarray: A zig zag signal.
        np.ndarray: The time array.
    """
    return np.arange(10), np.array([0, 5, 0, 5, 0, 5, 0, 5, 0, 5])

def test_threshold_crossing_interp_zig_zag_rising(arr_zig_zag):
    arr_t, arr_s = arr_zig_zag
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 2.5)
    assert len(arr_toa) == 5
    assert np.allclose(arr_toa, [0.5, 2.5, 4.5, 6.5, 8.5])

def test_threshold_crossing_interp_zig_zag_falling(arr_zig_zag):
    arr_t, arr_s = arr_zig_zag
    arr_toa = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 2.5, trigger_on_rising_edge=False)
    assert len(arr_toa) == 4
    assert np.allclose(arr_toa, [1.5, 3.5, 5.5, 7.5])

def test_wrong_n_est(arr_zig_zag):
    arr_t, arr_s = arr_zig_zag
    with pytest.raises(ValueError):
        triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 2.5, n_est=1)
    with pytest.raises(ValueError):
        triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 2.5, n_est=4)
    arr_toas_rising = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 2.5, n_est=5)
    assert len(arr_toas_rising) == 5
    assert np.allclose(arr_toas_rising, [0.5, 2.5, 4.5, 6.5, 8.5])

    with pytest.raises(ValueError):
        triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 2.5, n_est=1, trigger_on_rising_edge=False)
    with pytest.raises(ValueError):
        triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 2.5, n_est=3, trigger_on_rising_edge=False)
    arr_toas_falling = triggering_criteria.threshold_crossing_interp(arr_t, arr_s, 2.5, n_est=4, trigger_on_rising_edge=False)
    assert len(arr_toas_falling) == 4
    assert np.allclose(arr_toas_falling, [1.5, 3.5, 5.5, 7.5])

def test_hysteresis_rising_on_simple_ramp(arr_simple_ramp_up):
    arr_t, arr_s = arr_simple_ramp_up
    arr_toas = triggering_criteria.threshold_crossing_hysteresis_pos(
        arr_t,
        arr_s,
        5,
        2,
        1
    )
    assert len(arr_toas) == 1
    assert np.allclose(arr_toas, 5.0)
    ...


# Test threshold_crossing_interp todo:
# - Test with n_est, ensure exception is raised if 
#    n_est is less than the number of ToAs
# - Test with falling edge triggering
# - Test with n_est and falling edge triggering
# - ...

    
