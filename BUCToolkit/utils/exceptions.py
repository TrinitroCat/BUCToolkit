class FatalError(RuntimeError):
    """An unrecoverable error that requires aborting the current task."""


class IterationStuckError(RuntimeError):
    """Raised when an iterative algorithm cannot make a meaningful update."""
