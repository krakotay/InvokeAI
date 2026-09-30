from unittest.mock import patch

import pytest
import torch
from torch import nn

from invokeai.backend.anima.compile import compiled_anima_blocks


def _transformer():
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    return model


def test_compilation_preserves_modules_parameters_and_forward_hooks():
    model = _transformer()
    blocks = list(model.blocks)
    parameters = dict(model.named_parameters())
    state_keys = list(model.state_dict())
    hook_calls = []
    blocks[0].register_forward_hook(lambda *args: hook_calls.append(True))
    x = torch.ones(1, 2)
    expected = blocks[0](x)
    hook_calls.clear()

    with patch("torch.compile", side_effect=lambda fn, **kwargs: lambda *a, **kw: fn(*a, **kw)) as compile_mock:
        with compiled_anima_blocks(model):
            assert list(model.blocks) == blocks
            assert dict(model.named_parameters()) == parameters
            assert list(model.state_dict()) == state_keys
            torch.testing.assert_close(blocks[0](x), expected)
            assert hook_calls == [True]
        assert compile_mock.call_count == 2
        assert compile_mock.call_args.kwargs == {
            "backend": "inductor",
            "fullgraph": True,
            "options": {"emulate_precision_casts": True},
        }
    assert all("forward" not in block.__dict__ for block in blocks)


def test_restores_existing_instance_forward_after_inference_failure():
    model = _transformer()

    def existing_forward(x):
        return x

    model.blocks[0].forward = existing_forward
    with patch("torch.compile", side_effect=lambda fn, **kwargs: lambda x: fn(x)):
        with pytest.raises(RuntimeError, match="inference failed"):
            with compiled_anima_blocks(model):
                raise RuntimeError("inference failed")
    assert model.blocks[0].forward is existing_forward
    assert "forward" not in model.blocks[1].__dict__


def test_partial_compilation_failure_restores_all_forwards_and_propagates():
    model = _transformer()
    with patch("torch.compile", side_effect=[lambda x: x, RuntimeError("compile failed")]):
        with pytest.raises(RuntimeError, match="compile failed"):
            with compiled_anima_blocks(model):
                pytest.fail("must not suppress compilation errors")
    assert all("forward" not in block.__dict__ for block in model.blocks)
