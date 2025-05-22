from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
import inspect
from pathlib import Path
import os
from ..schema.pl import *
import scvelo as scv
import cellrank as cr
from scmcp_shared.util import add_op_log, filter_args,forward_request,get_ads
from scmcp_shared.logging_config import setup_logger
from ..util import set_fig_path
from scmcp_shared.schema import AdataModel
logger = setup_logger()


pl_mcp = FastMCP("CellrankMCP-Kernel-Server")


@pl_mcp.tool()
async def kernel_projection(
    request: KernelPlotProjectionModel, 
    adinfo: AdataModel = AdataModel()
    ):
    """Plot transition matrix as a stream or grid plot for a specified kernel."""
    try:
        result = await forward_request("kernel_projection", request, adinfo)
        if result is not None:
            return result 
        kernel_type = request.kernel
        ads = get_ads()
        kernel = ads.cr_kernel[kernel_type]
        
        # Check if transition matrix has been computed
        if not hasattr(kernel, 'transition_matrix') or kernel.transition_matrix is None:
            error_msg = f"Transition matrix for kernel '{kernel_type}' has not been computed. Please compute transition matrix first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Filter arguments based on the plot_projection method
        kwargs = filter_args(request, kernel.plot_projection)
        kwargs["save"] = kernel_type
        kwargs["show"] = False
        kwargs["color"] = request.color
        kwargs["legend_loc"] = request.legend_loc
        # Plot the projection
        kernel.plot_projection(**kwargs)
        kwargs["kernel"] = kernel_type
        fig_path = set_fig_path("scvelo_projection", **kwargs)
        return {"figpath": fig_path}
        
    except ToolError as e:
        raise ToolError(e)
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise ToolError(e.__context__)
        else:
            raise ToolError(e)


@pl_mcp.tool()
async def circular_projection(
    request: CircularProjectionModel,
    adinfo: AdataModel = AdataModel()
):
    """
    Visualize fate probabilities in a circular embedding. compute_fate_probabilities first.
    """
    try:
        result = await forward_request("circular_projection", request, adinfo)
        if result is not None:
            return result   
        # Check if AnnData object exists in the session
        ads = get_ads()
        adata = ads.get_adata(adinfo=adinfo)

        if "lineages_fwd" not in adata.obsm:
            raise ValueError("No lineages_fwd found. Please call compute_fate_probabilities first.")
        
        # Check if estimator exists and has computed fate probabilities
        kernel_types = []
        kernel_types = list(ads.cr_estimator.keys())
                
        # Filter arguments based on the circular_projection function
        kwargs = filter_args(request, cr.pl.circular_projection)
        
        kwargs["save"] = "circular_projection.png"
        kwargs["legend_loc"] = "right"
        kwargs["keys"] = request.keys
        # Call the circular_projection function
        cr.pl.circular_projection(adata, **kwargs)
        del kwargs["save"]
        fig_path = set_fig_path("circular_projection", **kwargs)
        return {
            "status": "success",
            "message": "Successfully created circular projection plot",
            "figpath": fig_path
        }
    except ToolError as e:
        raise ToolError(e)
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise ToolError(e.__context__)
        else:
            raise ToolError(e)

