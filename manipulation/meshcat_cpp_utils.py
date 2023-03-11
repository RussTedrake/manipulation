import warnings

from manipulation.meshcat_utils import *  # noqa

warnings.warn(
    "manipulation.meshcat_cpp_utils has been renamed to manipulation.meshcat_utils. This shim will be removed after 2022-12-31.",
    DeprecationWarning,
    stacklevel=2,
)
