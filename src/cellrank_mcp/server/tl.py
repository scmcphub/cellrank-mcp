from fastmcp import FastMCP, Context
import os
import scanpy as sc
from ..schema.tl import *
from scmcp_shared.util import filter_args, add_op_log, forward_request
from scmcp_shared.logging_config import setup_logger
logger = setup_logger()

tl_mcp = FastMCP("CellrankMCP-TL-Server")



@tl_mcp.tool()
async def tsne(
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for analysis"),
    request: TSNEModel = TSNEModel() 
):
    """t-distributed stochastic neighborhood embedding (t-SNE) for visualization"""

    try:
        result = await forward_request("tl_tsne", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result
        func_kwargs = filter_args(request, sc.tl.tsne)
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)
        sc.tl.tsne(adata, **func_kwargs)
        add_op_log(adata, sc.tl.tsne, func_kwargs)
        return [
            {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata},
        ]
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e 


@tl_mcp.tool()
async def umap(
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for analysis"),
    request: UMAPModel = UMAPModel() 
):
    """Uniform Manifold Approximation and Projection (UMAP) for visualization"""

    try:
        result = await forward_request("tl_umap", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result
        func_kwargs = filter_args(request, sc.tl.umap)
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)
        sc.tl.umap(adata, **func_kwargs)
        add_op_log(adata, sc.tl.umap, func_kwargs)
        return [
            {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata},
        ]
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e


@tl_mcp.tool()
async def diffmap(
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for analysis"),
    request: DiffMapModel = DiffMapModel() 
):
    """Diffusion Maps for dimensionality reduction"""

    try:
        result = await forward_request("tl_diffmap", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result    
        func_kwargs = filter_args(request, sc.tl.diffmap)
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)
        sc.tl.diffmap(adata, **func_kwargs)
        adata.obsm["X_diffmap"] = adata.obsm["X_diffmap"][:,1:]
        add_op_log(adata, sc.tl.diffmap, func_kwargs)
        return [
            {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata},
        ]
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e 


@tl_mcp.tool()
async def leiden(
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for analysis"),
    request: LeidenModel = LeidenModel() 
):
    """Leiden clustering algorithm for community detection"""

    try:
        result = await forward_request("tl_leiden", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result            
        func_kwargs = filter_args(request, sc.tl.leiden)
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)
        sc.tl.leiden(adata, **func_kwargs)
        add_op_log(adata, sc.tl.leiden, func_kwargs)
        return [
            {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata},
        ]
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e


@tl_mcp.tool()
async def louvain(
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for analysis"),
    request: LouvainModel = LouvainModel() 
):
    """Louvain clustering algorithm for community detection"""

    try:
        result = await forward_request("tl_louvain", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result          
        func_kwargs = filter_args(request, sc.tl.louvain)
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)
        sc.tl.louvain(adata, **func_kwargs)
        add_op_log(adata, sc.tl.louvain, func_kwargs)
        return [
            {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata},
        ]
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e




@tl_mcp.tool()
async def dpt(
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for analysis"),
    request: DPTModel = DPTModel() 
):
    """Diffusion Pseudotime (DPT) analysis"""

    try:
        result = await forward_request("tl_dpt", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result          
        func_kwargs = filter_args(request, sc.tl.dpt)
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)
        sc.tl.dpt(adata, **func_kwargs)
        add_op_log(adata, sc.tl.dpt, func_kwargs)
        return [
            {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata},
        ]
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e
