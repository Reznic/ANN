"""ANN errors module."""

class AnnError(BaseException):
    """Abstract ANN Error."""
    def __init__(self, message=""):
        self.message = self.__doc__ + "\n" + message

class EmptyNet(AnnError):
    """Network initialized with no layers."""
    pass

class LayerInputSizeError(AnnError):
    """Layer received invalid size input."""
    pass
