"""
This module updates the the sys.path.

The sys.path variable is a global variable, so importing this  module, will
change the path for the importing file.

This implies the directory structure:
    /somepath/gym-grasping
    /somepath/IRR
    /somepath/flownet2
"""
import os
import sys

dirname = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.abspath(os.path.join(dirname, "../../"))
if module_dir not in sys.path:
    sys.path.append(module_dir)
