class EpisodeDoneError(TimeoutError):
    """An error for when the episode is over."""
    pass


class InvalidActionError(ValueError):
    """An error for when an invalid action is taken"""
    pass