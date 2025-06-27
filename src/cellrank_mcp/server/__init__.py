from scmcp_shared.server.preset import (
    io_mcp,
    # ScanpyPreprocessingMCP,
    ScanpyToolsMCP,
    # ScanpyPlottingMCP,
    ScanpyUtilMCP,
)
from scmcp_shared.mcp_base import BaseMCPManager

from .preset.pl import pl_mcp
from .preset.kernel import kernel_mcp
from .preset.pp import pp_mcp
from .preset.estimator import estimator_mcp

# from .code.rag import rag_mcp
from scmcp_shared.server.code import nb_mcp

tl_mcp = ScanpyToolsMCP().mcp
ul_mcp = ScanpyUtilMCP(
    include_tools=["query_op_log", "check_samples"],
).mcp


class CellrankMCPManager(BaseMCPManager):
    """Manager class for Cellrank MCP modules."""

    def init_mcp(self):
        """Initialize available Cellrank MCP modules."""
        self.available_modules = {
            "io": io_mcp,
            "pp": pp_mcp,
            "kernel": kernel_mcp,
            "estimator": estimator_mcp,
            "pl": pl_mcp,
            "tl": tl_mcp,
            "ul": ul_mcp,
            # "rag": rag_mcp,
            "nb": nb_mcp,
        }
