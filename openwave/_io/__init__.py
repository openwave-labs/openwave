"""
I/O Module - Input/Output interfaces for OpenWave.

This module provides interfaces for:
- CLI (Command Line Interface)
- File Export (future)
- Video Management (future)
"""

from openwave._io.cli import cli_main

__all__ = ["cli_main"]
