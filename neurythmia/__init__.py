# ---INFO-----------------------------------------------------------------------
# Author(s):       Aditya Prakash
# Last Modified:   2023-06-22

# --Needed functionalities


# ---CONSTANTS------------------------------------------------------------------
"""
NeuRythmia version information.

We use semantic versioning (see http://semver.org/) and conform to PEP440 (see
https://www.python.org/dev/peps/pep-0440/). Release versions are git
tagged with the version.
"""
__version__ = (2, 0, 1)


# ---DEPENDENCIES---------------------------------------------------------------
from . import utils
from . import pipelines
from . import transformations
from . import forge
