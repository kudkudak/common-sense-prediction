"""
Config for project
"""

import os

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
ACL_ROOT_DIR = os.environ.get("ACL_ROOT_DIR", os.path.join(os.path.dirname(__file__), "ACL_CKBC"))