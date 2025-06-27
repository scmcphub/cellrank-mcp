import pytest


@pytest.fixture
def mcp():
    from cellrank_mcp.server import CellrankMCPManager
    from scmcp_shared.backend import AdataManager

    mcp = CellrankMCPManager("cellrank-mcp", backend=AdataManager).mcp
    return mcp
