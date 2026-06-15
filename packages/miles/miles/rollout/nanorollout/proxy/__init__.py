__all__ = ["TITOProxy"]


def __getattr__(name: str):
    if name == "TITOProxy":
        from .tito_server import TITOProxy

        return TITOProxy
    raise AttributeError(name)
