"""Pytest fixtures shared by the regression tests."""

from typing import Iterator

import pytest
import torch


def _reset_dynamo() -> None:
    """Clear Dynamo's process-wide compilation cache when it is available.

    Return:
        None.
    """
    reset = getattr(getattr(torch, '_dynamo', None), 'reset', None)
    if reset is not None:
        reset()


@pytest.fixture(autouse=True)
def _isolate_compiler_cache() -> Iterator[None]:
    """Isolate torch.compile state between regression tests.

    Dynamo's cache is process-wide. Compiled constraint Jacobians from one test
    otherwise consume the recompile budget of later constrained-MD tests, which
    makes the full suite fail even though each test passes in isolation.

    Return:
        A fixture generator that resets the cache before and after each test.
    """
    _reset_dynamo()
    yield
    _reset_dynamo()
