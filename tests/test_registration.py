import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from costream import synthetic
from costream.reactive.maps import TactileMaps, erode_contact, gradient_to_normal
from costream.reactive.registration import InsufficientOverlapError, register

MM = synthetic.MM_PER_PIXEL


def _yaw_deg(T):
    return np.degrees(Rotation.from_matrix(T[:3, :3]).as_rotvec()[2])


@pytest.mark.parametrize('dx, dy, yaw_deg', [(.3, -.2, 0.), (0., 0., 4.), (.25, .15, -3.)])
def test_recovers_in_plane_slip_and_pins_sign_convention(dx, dy, yaw_deg):
    # S1_T_S0 maps reference contact points into the current frame: its translation is the contact's shift.
    ref = synthetic.tactile_maps()
    cur = synthetic.tactile_maps(dx_mm=dx, dy_mm=dy, yaw=np.radians(yaw_deg))
    T = register(ref, cur, mm_per_pixel=MM, rng=np.random.default_rng(0))
    np.testing.assert_allclose(T[:2, 3] * 1000, [dx, dy], atol=.01)
    assert _yaw_deg(T) == pytest.approx(yaw_deg, abs=.05)
    assert abs(T[2, 3]) * 1000 < .01


def test_identical_frames_give_identity():
    maps = synthetic.tactile_maps()
    np.testing.assert_allclose(register(maps, maps, mm_per_pixel=MM, n_samples=None), np.eye(4), atol=1e-9)


def test_disjoint_contact_raises():
    with pytest.raises(InsufficientOverlapError):
        register(synthetic.tactile_maps(), synthetic.tactile_maps(dx_mm=12.), mm_per_pixel=MM)


def test_reference_without_contact_raises():
    with pytest.raises(InsufficientOverlapError):
        register(synthetic.tactile_maps(pressed=False), synthetic.tactile_maps(), mm_per_pixel=MM)


def test_validates_scale_and_size():
    maps = synthetic.tactile_maps()
    with pytest.raises(ValueError, match='mm_per_pixel'):
        register(maps, maps, mm_per_pixel=0.)
    with pytest.raises(ValueError, match='size'):
        register(maps, synthetic.tactile_maps(shape=(90, 120)), mm_per_pixel=MM)


def test_seeded_sampling_is_reproducible():
    ref, cur = synthetic.tactile_maps(), synthetic.tactile_maps(dx_mm=.2)
    first = register(ref, cur, mm_per_pixel=MM, rng=np.random.default_rng(3))
    second = register(ref, cur, mm_per_pixel=MM, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(first, second)


def test_maps_validation_and_helpers():
    with pytest.raises(ValueError, match='boolean'):
        TactileMaps(np.zeros((4, 4, 3)), np.zeros((4, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError, match='one size'):
        TactileMaps(np.zeros((4, 4, 3)), np.zeros((4, 5)), np.zeros((4, 4), bool))
    np.testing.assert_allclose(gradient_to_normal(np.zeros((3, 3, 2)))[..., 2], 1.)
    contact = np.zeros((96, 96), bool)
    contact[10:60, 10:60] = True
    eroded = erode_contact(contact)
    assert eroded.dtype == bool and not eroded[10, 10] and eroded[30, 30]


@pytest.mark.parametrize('init', [np.ones((3, 3)), np.full((4, 4), np.nan), np.diag([1., 1., -1., 1.])])
def test_rejects_invalid_initial_guess(init):
    maps = synthetic.tactile_maps()
    with pytest.raises(ValueError):
        register(maps, maps, mm_per_pixel=MM, init=init)


def test_sampler_matches_scipy_grid_constant_bilinear():
    from scipy.ndimage import map_coordinates
    from costream.reactive.registration import _padded, _sample
    rng = np.random.default_rng(0)
    image = rng.normal(size=(12, 17, 3))
    x = rng.uniform(-3, 20, size=200)
    y = rng.uniform(-3, 15, size=200)
    expected = np.stack([map_coordinates(image[..., c], np.stack([y, x]), order=1, mode='grid-constant')
                         for c in range(3)], axis=-1)
    np.testing.assert_allclose(_sample(_padded(image), x, y), expected, atol=1e-12)
