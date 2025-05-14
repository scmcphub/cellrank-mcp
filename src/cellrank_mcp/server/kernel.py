from fastmcp import FastMCP, Context
import os
import inspect
from pathlib import Path
import scanpy as sc
import cellrank as cr
import scvelo as scv
from pathlib import Path
from ..schema.kernel import *
from scmcp_shared.util import filter_args,forward_request
from scmcp_shared.logging_config import setup_logger


logger = setup_logger()


kernel_mcp = FastMCP("CellrankMCP-Kernel-Server")


@kernel_mcp.tool()
async def create_kernel(
    request: KernelModel, 
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for processing")
):
    """Create a CellRank kernel based on the specified type and parameters."""
    try:
        # Check if AnnData object exists in the session
        result = await forward_request("kernel_create_kernel", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result
        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid).copy()
        kernel_type = request.kernel
        
        kernel_dic = {
            "pseudotime": cr.kernels.PseudotimeKernel,
            "cytotrace": cr.kernels.CytoTRACEKernel,
            "velocity": cr.kernels.VelocityKernel,
            "connectivity": cr.kernels.ConnectivityKernel,
            "realtime": cr.kernels.RealTimeKernel
        }
        if kernel_type not in kernel_dic:
            return {
                "status": "error",
                "message": f"不支持的kernel类型: {kernel_type}"
            }
        kernel_class = kernel_dic[kernel_type]
        kwargs = filter_args(request, kernel_class)
        if kernel_type == "cytotrace":
            if "Ms" not in adata.layers or "Mu" not in adata.layers:
                scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
            kernel = kernel_class(adata=adata, **kwargs).compute_cytotrace()
        else:
            kernel = kernel_class(adata=adata, **kwargs)
        
        ads.cr_kernel[kernel_type] = kernel
        ads.set_adata(adata, sampleid=sampleid, sdtype=dtype)
        return {
            "status": "success",
            "message": f"Successfully created {kernel_type} kernel",
            "kernel_type": kernel_type
        }
    
    except Exception as e:
        logger.error(f"Error creating kernel: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to create kernel: {str(e)}"
        }


@kernel_mcp.tool()
async def compute_transition_matrix(
    request: ComputeTransitionMatrixModel, 
    ctx: Context,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for processing")
    ):
    """Compute transition matrix for a specified kernel using appropriate parameters."""
    try:
        result = await forward_request("compute_transition_matrix", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result
        kernel_type = request.kernel
        ads = ctx.request_context.lifespan_context
        kernel = ads.cr_kernel[kernel_type].copy()        
        kwargs = filter_args(request, kernel.compute_transition_matrix)
        kwargs["show_progress_bar"] = False
        kernel.compute_transition_matrix(**kwargs)
        ads.cr_kernel[kernel_type] = kernel
        return {
            "status": "success",
            "message": f"Successfully computed transition matrix for {kernel_type} kernel"
        }
        
    except Exception as e:
        logger.error(f"Error computing transition matrix: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to compute transition matrix: {str(e)}"
        }

