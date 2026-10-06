import sys
import types

import numpy as np
import pytest

from costream.adapters.gelsight import GsSdkReconstructor


class _FakeReconstructor:
    def __init__(self, model_path, contact_mode='standard', device='cpu'):
        self.args = (model_path, contact_mode, device)
        self.background = None

    def load_bg(self, image):
        self.background = image

    def get_surface_info(self, image, ppmm):
        self.ppmm = ppmm
        gradient = np.zeros(image.shape[:2] + (2,))
        gradient[..., 0] = .5
        return gradient, np.ones(image.shape[:2]), np.ones(image.shape[:2], dtype=np.uint8)


@pytest.fixture
def fake_gs_sdk(monkeypatch):
    package = types.ModuleType('gs_sdk')
    module = types.ModuleType('gs_sdk.gs_reconstruct')
    module.Reconstructor = _FakeReconstructor
    package.gs_reconstruct = module
    monkeypatch.setitem(sys.modules, 'gs_sdk', package)
    monkeypatch.setitem(sys.modules, 'gs_sdk.gs_reconstruct', module)


def test_adapter_wraps_gs_sdk_output_as_tactile_maps(fake_gs_sdk):
    background = np.zeros((6, 8, 3), np.uint8)
    reconstructor = GsSdkReconstructor('model.pth', background, mm_per_pixel=.0634, device='cuda')
    maps = reconstructor.reconstruct(np.zeros((6, 8, 3), np.uint8))
    assert reconstructor._recon.args == ('model.pth', 'standard', 'cuda')
    assert reconstructor._recon.background is background and reconstructor._recon.ppmm == .0634
    assert maps.contact.dtype == bool and maps.contact.all()
    np.testing.assert_allclose(maps.normal[0, 0], np.array([-.5, 0., 1.]) / np.sqrt(1.25))


def test_adapter_requires_a_positive_scale(fake_gs_sdk):
    with pytest.raises(ValueError, match='mm_per_pixel'):
        GsSdkReconstructor('model.pth', np.zeros((2, 2, 3)), mm_per_pixel=0.)


def test_missing_gs_sdk_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gs_sdk', None)
    with pytest.raises(ImportError, match='github.com/joehjhuang/gs_sdk'):
        GsSdkReconstructor('model.pth', np.zeros((2, 2, 3)), mm_per_pixel=.0634)
