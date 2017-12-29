"""
Config for project
"""

from __future__ import print_function

import os

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
ACL_ROOT_DIR = os.environ.get("ACL_ROOT_DIR", os.path.join(os.path.dirname(__file__), "ACL_CKBC"))

# A bit risky for some packages, but setting default logging to higher level
import logging
logging.getLogger('').setLevel(logging.INFO)