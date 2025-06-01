"""
Command-line interface for cellrank-mcp.

This module provides a CLI entry point for the cellrank-mcp package.
"""

from scmcp_shared.cli import MCPCLI
from .server import CellrankMCPManager

cli = MCPCLI(
    name="cellrank-mcp", 
    help_text="Cellrank MCP Server CLI",
    manager=CellrankMCPManager
)
