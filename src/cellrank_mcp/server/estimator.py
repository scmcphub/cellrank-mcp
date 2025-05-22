from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
import cellrank as cr
import os
from pathlib import Path
import numpy as np
from ..schema.estimator import *
from scmcp_shared.util import filter_args, add_op_log, forward_request, get_ads
from scmcp_shared.logging_config import setup_logger
from scmcp_shared.schema import AdataModel
logger = setup_logger()

estimator_mcp = FastMCP("CellrankMCP-Estimator-Server")


@estimator_mcp.tool()
async def create_and_fit_gpcca(
    request: GPCCAFitModel,
    adinfo: AdataModel = AdataModel()
):
    """Generalized Perron Cluster Cluster Analysis (GPCCA), Use it to compute macrostates.
    Need to compute transition matrix first as `.compute_transition_matrix()
    """
    try:
        result = await forward_request("create_and_fit_gpcca", request, adinfo)
        if result is not None:
            return result    
        kernel_type = request.kernel

        ads = get_ads()
        adata = ads.get_adata(adinfo=adinfo)

        kernel = ads.cr_kernel[kernel_type]    
        estimator = cr.estimators.GPCCA(kernel)
        kwargs = filter_args(request, estimator.fit)
        schur_kwargs = filter_args(request, cr.estimators.GPCCA.compute_schur)
        kwargs.update(schur_kwargs)
        estimator.fit(**kwargs)
        ads.cr_estimator[kernel_type] = estimator
        return {
            "status": "success",
            "message": f"Successfully created and fitted the GPCCA estimator for kernel '{kernel_type}'",
            "estimator": estimator
        }
    except ToolError as e:
        raise ToolError(e)
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise ToolError(e.__context__)
        else:
            raise ToolError(e)


@estimator_mcp.tool()
async def predict_terminal_states(
    request: GPCCAPredictTerminalStatesModel, 
    adinfo: AdataModel = AdataModel()
):
    """
    Predict terminal states from macrostates. This automatically selects terminal states from the computed macrostates.
    """  
    try:
        result = await forward_request("predict_terminal_states", request, adinfo)
        if result is not None:
            return result  
        kernel_type = request.kernel
        ads = get_ads()
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
    except ToolError as e:
        raise ToolError(e)
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise ToolError(e.__context__)
        else:
            raise ToolError(e)


@estimator_mcp.tool()
async def predict_initial_states(
    request: GPCCAPredictInitialStatesModel, 
    adinfo: AdataModel = AdataModel()
):
    """
    Compute initial states from macrostates using coarse_stationary_distribution.
    """
    try:
        result = await forward_request("predict_initial_states", request, adinfo)
        if result is not None:
            return result
        ads = get_ads()
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
    except ToolError as e:
        raise ToolError(e)
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise ToolError(e.__context__)
        else:
            raise ToolError(e)


@estimator_mcp.tool()
async def compute_fate_probabilities(
    request: GPCCAComputeFateProbabilitiesModel, 
    adinfo: AdataModel = AdataModel()
):
    """Compute fate probabilities for cells.
    """
    try:
        result = await forward_request("compute_fate_probabilities", request, adinfo)
        if result is not None:
            return result  
        kernel_type = request.kernel
        ads = get_ads()
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
    except ToolError as e:
        raise ToolError(e)
    except Exception as e:
        if hasattr(e, '__context__') and e.__context__:
            raise ToolError(e.__context__)
        else:
            raise ToolError(e)