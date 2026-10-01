import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_only_full_load import (
    CachedModelOnlyFullLoad,
)
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_loaders.anima import AnimaCheckpointModel, _get_anima_num_blocks


@pytest.mark.parametrize(
    "required,compiled,expected", [(False, False, True), (True, False, False), (False, True, False)]
)
def test_anima_loading_policy(required, compiled, expected):
    loader = object.__new__(AnimaCheckpointModel)
    loader._app_config = SimpleNamespace(anima_require_full_vram=required, anima_compile_blocks=compiled)
    assert loader._allow_partial_loading() is expected


def test_full_residency_policy_does_not_disable_streaming_other_models():
    cache = ModelCache(
        execution_device_working_mem_gb=0.75,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
        execution_device="cpu",
        storage_device="cpu",
        logger=logging.getLogger(__name__),
    )
    try:
        with patch(
            "invokeai.backend.model_manager.load.model_cache.model_cache._has_dedicated_vram", return_value=True
        ):
            cache.put("anima", nn.Linear(2, 2), execution_device=torch.device("cuda:0"), allow_partial_loading=False)
            cache.put("other", nn.Linear(2, 2), execution_device=torch.device("cuda:0"))
        assert isinstance(cache._cached_models["anima"].cached_model, CachedModelOnlyFullLoad)
        assert isinstance(cache._cached_models["other"].cached_model, CachedModelWithPartialLoad)
    finally:
        cache.shutdown()


@pytest.mark.parametrize("depth", [28, 40])
def test_checkpoint_depth(depth):
    assert _get_anima_num_blocks({f"blocks.{i}.weight": None for i in range(depth)}) == depth


def test_checkpoint_depth_rejects_missing_blocks():
    with pytest.raises(ValueError, match="missing DiT block indices"):
        _get_anima_num_blocks({"blocks.0.weight": None, "blocks.2.weight": None})


def test_per_generation_full_residency_survives_unlock_for_whole_model_eviction():
    cache = ModelCache(
        execution_device="cpu",
        storage_device="cpu",
        logger=logging.getLogger(__name__),
        execution_device_working_mem_gb=0.75,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=True,
    )
    try:
        with patch(
            "invokeai.backend.model_manager.load.model_cache.model_cache._has_dedicated_vram", return_value=True
        ):
            cache.put("anima", nn.Linear(2, 2), execution_device=torch.device("cuda:0"))
        record = cache._cached_models["anima"]
        assert isinstance(record.cached_model, CachedModelWithPartialLoad)
        with patch.object(cache, "_load_locked_model"), patch.object(cache, "_log_cache_state"):
            cache.lock(record, None, force_full_load=True)
            # A nested default lock must not downgrade the active full-residency user.
            cache.lock(record, None)
            assert record.force_full_load
            cache.unlock(record)
            cache.unlock(record)
            with patch.object(record.cached_model, "full_load_to_vram", return_value=24) as load:
                assert cache._move_model_to_vram(record, 1) == 24
                load.assert_called_once()
            with patch.object(record.cached_model, "full_unload_from_vram", return_value=24) as unload:
                assert cache._move_model_to_ram(record, 1) == 24
                unload.assert_called_once()
            cache.lock(record, None)
            assert not record.force_full_load
            cache.unlock(record)
            with patch.object(record.cached_model, "partial_load_to_vram", return_value=1) as partial:
                cache._move_model_to_vram(record, 1)
                partial.assert_called_once_with(1)
    finally:
        cache.shutdown()
