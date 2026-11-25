"""The failures warpc reports by name instead of producing wrong numbers."""


class UnsupportedModel(ValueError):
    """The checkpoint asks for something the dense pipeline cannot reproduce; the message names each setting."""


class LoweringError(RuntimeError):
    """A graph has ops or states no capability of the backend can run; the message names them and the reasons."""
