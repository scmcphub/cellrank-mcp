from scmcp_shared.server import (
    BaseMCPManager, io_mcp,
    ScanpyPreprocessingMCP, 
    ScanpyToolsMCP,
    ScanpyPlottingMCP,
    ScanpyUtilMCP
)

from .pl import pl_mcp
from .kernel import kernel_mcp
from .pp import pp_mcp
from .estimator import estimator_mcp


tl_mcp = ScanpyToolsMCP().mcp
ul_mcp = ScanpyUtilMCP(
    include_tools=["query_op_log", "check_samples"],
).mcp


class CellrankMCPManager(BaseMCPManager):
    """Manager class for Cellrank MCP modules."""
    
    def _init_modules(self):
        """Initialize available Cellrank MCP modules."""
        self.available_modules = {
            "io": io_mcp, 
            "pp": pp_mcp, 
            "kernel": kernel_mcp, 
            "estimator": estimator_mcp, 
            "pl": pl_mcp, 
            "tl": tl_mcp,
            "ul": ul_mcp
}
