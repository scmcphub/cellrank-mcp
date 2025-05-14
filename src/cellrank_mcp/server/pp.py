from fastmcp import FastMCP, Context
import scvelo as scv
import scanpy as sc
import inspect
import os
from pathlib import Path
from ..schema.pp import *
from scmcp_shared.util import filter_args, add_op_log,forward_request
from scmcp_shared.logging_config import setup_logger

logger = setup_logger()

pp_mcp = FastMCP("CellrankMCP-Preprocessing-Server")


@pp_mcp.tool()
async def filter_and_normalize(
    request: FilterAndNormalizeModel, 
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for processing")
):
    """
    Filter and normalize AnnData object for velocity analysis.
    This function filters genes based on minimum number of counts and cells,
    normalizes the data, and identifies highly variable genes.
    """
    try:
        result = await forward_request("pp_filter_and_normalize", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result   
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid).copy()
        if 'spliced' in adata.layers:
            # Filter arguments based on the function parameters
            kwargs = filter_args(request, scv.pp.filter_and_normalize)
            if "log" in inspect.signature(scv.pp.filter_and_normalize).parameters:
                kwargs["log"] = False
            # Run filter_and_normalize
            scv.pp.filter_and_normalize(adata, **kwargs)            
            sc.pp.log1p(adata)
            add_op_log(adata, scv.pp.filter_and_normalize, kwargs)
            add_op_log(adata, sc.pp.log1p, {})
        else:
            kwargs = filter_args(request, sc.pp.filter_genes)
            sc.pp.filter_genes(adata, filter_args(request, sc.pp.filter_genes))
            sc.pp.normalize_total(adata, filter_args(request, sc.pp.normalize_total))
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata,  filter_args(request, sc.pp.highly_variable_genes))
            adata.layers["spliced"] = adata.X
            adata.layers["unspliced"] = adata.X
        ads.set_adata(adata, sampleid=sampleid, sdtype=dtype)
        return {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata}
        
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e


@pp_mcp.tool()
async def pca(
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for preprocessing"),
    request: PCAModel = PCAModel() 
):
    """Principal component analysis"""

    try:
        result = await forward_request("pp_pca", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result
        func_kwargs = filter_args(request, sc.pp.pca)
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)
        sc.pp.pca(adata, **func_kwargs)
        add_op_log(adata, sc.pp.pca, func_kwargs)
        return [
            {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata}
        ]
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e 


@pp_mcp.tool()
async def neighbors(
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for preprocessing"),
    request: NeighborsModel = NeighborsModel() 
):
    """Compute nearest neighbors distance matrix and neighborhood graph"""

    try:
        result = await forward_request("pp_neighbors", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result
        func_kwargs = filter_args(request, sc.pp.neighbors)
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)
        sc.pp.neighbors(adata, **func_kwargs)
        add_op_log(adata, sc.pp.neighbors, func_kwargs)
        return [
            {"sampleid": sampleid or ads.active_id, "dtype": dtype, "adata": adata}
        ]
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e
