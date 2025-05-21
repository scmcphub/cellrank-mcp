import asyncio
from fastmcp import FastMCP
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


from .pl import pl_mcp
from .kernel import kernel_mcp
from .pp import pp_mcp
from .estimator import estimator_mcp

import scmcp_shared.server as shs
from  scmcp_shared.util import filter_tools

ads = shs.AdataState()

@asynccontextmanager
async def adata_lifespan(server: FastMCP) -> AsyncIterator[Any]:
    yield ads

cellrank_mcp = FastMCP("Cellrank-MCP-Server", lifespan=adata_lifespan)

async def setup(modules=None):

    pp1_mcp = await filter_tools(
        shs.pp_mcp,
        include_tools=["neighbors"]
    )
    mcp_dic = {
        "io": shs.io_mcp, 
        "pp": pp1_mcp, 
        "kernel": kernel_mcp, 
        "estimator": estimator_mcp, 
        "pl": shs.pl_mcp, "tl": shs.tl_mcp
        }
    if modules is None or modules == "all":
        modules = mcp_dic.keys()
    for module in modules:
        await cellrank_mcp.import_server(module, mcp_dic[module])
    await cellrank_mcp.import_server("pl", pl_mcp)
    await cellrank_mcp.import_server("pp", pp_mcp)

