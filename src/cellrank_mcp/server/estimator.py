from fastmcp import FastMCP, Context
import cellrank as cr
import os
from pathlib import Path
import numpy as np
from ..schema.estimator import *
from scmcp_shared.util import filter_args, add_op_log, forward_request
from scmcp_shared.logging_config import setup_logger

logger = setup_logger()

estimator_mcp = FastMCP("CellrankMCP-Estimator-Server")


@estimator_mcp.tool()
async def create_and_fit_gpcca(
    ctx: Context,
    request: GPCCAFitModel,
    dtype: str = Field(default="exp", description="the datatype of anndata.X"),
    sampleid: str = Field(default=None, description="adata sampleid for processing")
):
    """Generalized Perron Cluster Cluster Analysis (GPCCA), Use it to compute macrostates.
    Need to compute transition matrix first as `.compute_transition_matrix()
    """
    try:
        result = await forward_request("create_and_fit_gpcca", request, sampleid=sampleid, dtype=dtype)
        if result is not None:
            return result    
        kernel_type = request.kernel

        ads = ctx.request_context.lifespan_context
        adata = ads.get_adata(dtype=dtype, sampleid=sampleid)

        kernel = ads.cr_kernel[kernel_type]    
        estimator = cr.estimators.GPCCA(kernel)
        kwargs = filter_args(request, estimator.fit)
        schur_kwargs = filter_args(request, cr.estimators.GPCCA.compute_schur)
        kwargs.update(schur_kwargs)
        estimator.fit(**kwargs)
        ads.cr_estimator[kernel_type] = estimator
        return {
            "status": "success",
            "message": f"成功创建并拟合了内核 '{kernel_type}' 的 GPCCA 估计器",
            "estimator": estimator
        }
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e


@estimator_mcp.tool()
async def predict_terminal_states(
    ctx: Context,
    request: GPCCAPredictTerminalStatesModel, 
    dtype: str = None,
    sampleid: str = None
):
    """
    Predict terminal states from macrostates. This automatically selects terminal states from the computed macrostates.
    """  
    try:
        result = await forward_request("predict_terminal_states", request)
        if result is not None:
            return result  
        kernel_type = request.kernel
        ads = ctx.request_context.lifespan_context
        estimator = ads.cr_estimator[kernel_type].copy()
        
        # Filter arguments based on the predict_terminal_states method
        kwargs = filter_args(request, estimator.predict_terminal_states)
        
        # Predict terminal states
        estimator.predict_terminal_states(**kwargs)
        
        # Get information about the terminal states
        terminal_states = estimator.terminal_states.cat.categories if hasattr(estimator, 'terminal_states') else []
        ads.cr_estimator[kernel_type] = estimator
        return {
            "status": "success",
            "estimator": estimator,
            "kernel_type": kernel_type,
            "terminal_states": terminal_states
        }
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e


@estimator_mcp.tool()
async def predict_initial_states(
    ctx: Context,
    request: GPCCAPredictInitialStatesModel, 
    dtype: str = None,
    sampleid: str = None
):
    """
    Compute initial states from macrostates using coarse_stationary_distribution.
    """
    try:
        result = await forward_request("predict_initial_states", request)
        if result is not None:
            return result    
        estimator = ads.cr_estimator[kernel_type].copy()
        # Filter arguments based on the predict_initial_states method
        kwargs = filter_args(request, estimator.predict_initial_states)
        
        # Predict initial states
        estimator.predict_initial_states(**kwargs)
        
        # Get information about the initial states
        initial_states = estimator.initial_states.cat.categories if hasattr(estimator, 'initial_states') else []
        ads.cr_estimator[kernel_type] = estimator 
        return {
            "status": "success",
            "message": f"Successfully predicted initial states for kernel '{kernel_type}'",
            "kernel_type": kernel_type,
            "initial_states": initial_states
        }
        
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e


@estimator_mcp.tool()
async def compute_fate_probabilities(
    request: GPCCAComputeFateProbabilitiesModel, 
    ctx: Context,
    dtype: str = None,
    sampleid: str = None
):
    """Compute fate probabilities for cells.
    """
    try:
        result = await forward_request("compute_fate_probabilities", request)
        if result is not None:
            return result  
        kernel_type = request.kernel
        ads = ctx.request_context.lifespan_context
        estimator = ads.cr_estimator[kernel_type].copy()
        
        # Check if terminal states have been computed
        if not hasattr(estimator, 'terminal_states') or estimator.terminal_states is None:
            raise ValueError("Terminal states have not been computed. Please run predict_terminal_states first.")
        
        # Filter arguments based on the compute_fate_probabilities method
        kwargs = filter_args(request, estimator.compute_fate_probabilities)
        kwargs["show_progress_bar"] = False
        # Compute fate probabilities
        estimator.compute_fate_probabilities(**kwargs)
        
        # Get information about the fate probabilities
        fate_probs_computed = hasattr(estimator, 'fate_probabilities')
        terminal_states = estimator.terminal_states.cat.categories.tolist() if hasattr(estimator, 'terminal_states') else []
        ads.cr_estimator[kernel_type] = estimator 
        return {
            "status": "success",
            "message": f"Successfully computed fate probabilities for kernel '{kernel_type}'",
            "kernel_type": kernel_type,
            "terminal_states": terminal_states,
            "fate_probabilities_computed": fate_probs_computed,
            "estimator": estimator,
        }
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise Exception(f"{str(e.__context__)}")
        else:
            raise e
