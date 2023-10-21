import warnings

from pydrake.gym import *  # noqa

warnings.warn(
    "manipulation.drake_gym has moved to pydrake.gym. This shim will be removed after 2023-12-31.",
    DeprecationWarning,
    stacklevel=2,
)
