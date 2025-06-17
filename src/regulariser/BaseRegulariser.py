from abc import ABC, abstractmethod


class BaseRegulariser(ABC):
    def __init__(self, **kwargs: dict) -> None:
        """
        Base class for regularisers.
        :param kwargs: Additional keyword arguments.
        """
        self.kwargs = kwargs

    @abstractmethod
    def __call__(self, **kwargs: dict) -> float:
        raise NotImplementedError(
            "The __call__ method must be implemented in the subclass."
        )
