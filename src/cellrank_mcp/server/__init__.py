import asyncio
from fastmcp import FastMCP
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from scmcp_shared.server import io_mcp
from.pl import pl_mcp
from.tl import tl_mcp
from.kernel import kernel_mcp
from.pp import pp_mcp
from.estimator import estimator_mcp



class AdataState:
    def __init__(self):
        self.adata_dic ={"exp": {}, "splicing": {}}
        self.active_id = None
        self.metadata = {}
        self.cr_kernel = {}
        self.cr_estimator = {}
    def get_adata(self, sampleid=None, dtype="exp"):
        try:
            if self.active_id is None:
                return None
            sampleid = sampleid or self.active_id
            return self.adata_dic[dtype][sampleid]
        except KeyError as e:
            raise KeyError(f"Key {e} not found in adata_dic")
        except Exception as e:
            raise Exception(f"Error: {e}")
    
    def set_adata(self, adata, sampleid=None, sdtype="exp"):
        sampleid = sampleid or self.active_id
        self.adata_dic[sdtype][sampleid] = adata


ads = AdataState()

@asynccontextmanager
async def adata_lifespan(server: FastMCP) -> AsyncIterator[Any]:
    yield ads

cellrank_mcp = FastMCP("Cellrank-MCP-Server", lifespan=adata_lifespan)

async def setup(modules=None):
    mcp_dic = {
        "io": io_mcp, 
        "pp": pp_mcp, 
        "kernel": kernel_mcp, 
        "estimator": estimator_mcp, 
        "pl": pl_mcp, "tl": tl_mcp
        }
    if modules is None or modules == "all":
        modules = ["io", "pp", "kernel", "estimator", "pl", "tl"]
    for module in modules:
        await cellrank_mcp.import_server(module, mcp_dic[module])

