import scvelo as scv
import scanpy as sc
import inspect
from ...schema.pp import *
from scmcp_shared.util import filter_args, add_op_log, forward_request, get_ads
from scmcp_shared.logging_config import setup_logger
from fastmcp.exceptions import ToolError
from scmcp_shared.schema import AdataInfo
from scmcp_shared.server.preset import ScanpyPreprocessingMCP


logger = setup_logger()


pp_mcp = ScanpyPreprocessingMCP(include_tools=["neighbors"]).mcp


@pp_mcp.tool()
def filter_and_normalize(
    request: FilterAndNormalizeModel, adinfo: AdataInfo = AdataInfo()
):
    """
    Preprocess data: filter and normalize AnnData object for velocity/pseudotime analysis.
    """
    try:
        result = forward_request("filter_and_normalize", request, adinfo)
        if result is not None:
            return result
        ads = get_ads()
        adata = ads.get_adata(adinfo=adinfo).copy()
        if "spliced" in adata.layers:
            # Filter arguments based on the function parameters
            kwargs = filter_args(request, scv.pp.filter_and_normalize)
            if "log" in inspect.signature(scv.pp.filter_and_normalize).parameters:
                kwargs["log"] = False
            # Run filter_and_normalize
            scv.pp.filter_and_normalize(adata, **kwargs)
            sc.pp.log1p(adata)
            add_op_log(adata, scv.pp.filter_and_normalize, kwargs, adinfo)
            add_op_log(adata, sc.pp.log1p, {}, adinfo)
        else:
            sc.pp.filter_genes(adata, **filter_args(request, sc.pp.filter_genes))
            sc.pp.normalize_total(adata, **filter_args(request, sc.pp.normalize_total))
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(
                adata, **filter_args(request, sc.pp.highly_variable_genes)
            )
            adata.layers["spliced"] = adata.X
            adata.layers["unspliced"] = adata.X
        ads.set_adata(adata, adinfo=adinfo)
        return {
            "sampleid": adinfo.sampleid or ads.active_id,
            "dtype": adinfo.adtype,
            "adata": adata,
        }

    except ToolError as e:
        raise ToolError(e)
    except Exception as e:
        if hasattr(e, "__") and e.__:
            raise ToolError(e.__context__)
        else:
            raise ToolError(e)
