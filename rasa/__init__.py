import logging
import warnings
import os

# Suppress deprecation warnings from external packages
warnings.filterwarnings(
    "ignore", 
    message=".*pkg_resources is deprecated.*", 
    category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*Deprecated call to.*declare_namespace.*",
    category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*jax.xla_computation is deprecated.*",
    category=DeprecationWarning
)

# Set environment variable to silence SQLAlchemy warnings as well
os.environ.setdefault("SQLALCHEMY_SILENCE_UBER_WARNING", "1")

try:
    # Suppress SQLAlchemy 2.0 compatibility warnings
    from sqlalchemy import exc as sqla_exc
    warnings.filterwarnings(
        "ignore",
        category=sqla_exc.MovedIn20Warning
    )
except ImportError:
    pass

from rasa import version, plugin  # noqa: F401
from rasa.api import run, train, test  # noqa: F401

# define the version before the other imports since these need it
__version__ = version.__version__


logging.getLogger(__name__).addHandler(logging.NullHandler())
