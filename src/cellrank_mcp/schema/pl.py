from pydantic import (
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
    BaseModel
)
from typing import Optional, List, Dict, Union, Literal, Tuple, Any



class KernelPlotProjectionModel(BaseModel):
    """Input schema for plotting transition matrix as a stream or grid plot in CellRank kernels."""
    
    basis: str = Field(
        default='umap',
        description="Key in obsm containing the basis."
    )
    
    key_added: Optional[str] = Field(
        default=None,
        description="If not None, save the result to adata.obsm['{key_added}']. Otherwise, save the result to 'T_fwd_{basis}' or 'T_bwd_{basis}', depending on the direction."
    )
    
    recompute: bool = Field(
        default=False,
        description="Whether to recompute the projection if it already exists."
    )
    
    stream: bool = Field(
        default=True,
        description="If True, use velocity_embedding_stream(). Otherwise, use velocity_embedding_grid()."
    )
    
    connectivities: Optional[Any] = Field(
        default=None,
        description="Connectivity matrix to use for projection. If None, use ones from the underlying kernel, if possible."
    )
    
    color: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Key for annotations of observations/cells or variables/genes."
    )
    
    legend_loc: str = Field(
        default=None,
        description="Location of legend, either 'on data', 'right margin' or valid keywords for matplotlib.legend."
    )
    
    kernel: Literal['pseudotime', 'cytotrace', 'velocity', 'connectivity', 'realtime'] = Field(
        description="Type of kernel to use."
    )


class CircularProjectionModel(BaseModel):
    """
    Input schema for CellRank's circular_projection function which visualizes fate probabilities in a circular embedding.
    """
    
    keys: Union[str, List[str]] = Field(
        description="Keys in obs or var_names. Can include 'kl_divergence' or 'entropy'."
    )
    
    backward: bool = Field(
        default=False,
        description="Direction of the process."
    )
    
    lineages: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Lineages to plot. If None, plot all lineages."
    )
    
    early_cells: Optional[Union[Dict[str, List[str]], List[str]]] = Field(
        default=None,
        description="Cell ids or a mask marking early cells used to define the average fate probabilities. If None, use all cells."
    )
    
    lineage_order: Optional[Literal['default', 'optimal']] = Field(
        default=None,
        description="How to order lineages. 'optimal' solves the Traveling salesman problem. 'default' uses the order specified by lineages."
    )
    
    metric: Union[str, Any] = Field(
        default='correlation',
        description="Metric to use when constructing pairwise distance matrix when lineage_order = 'optimal'."
    )
    
    normalize_by_mean: bool = Field(
        default=True,
        description="If True, normalize each lineage by its mean probability."
    )
    
    ncols: int = Field(
        default=4,
        description="Number of columns when plotting multiple keys."
    )
    
    space: float = Field(
        default=0.25,
        description="Horizontal and vertical space between for subplots_adjust()."
    )
    
    use_raw: bool = Field(
        default=False,
        description="Whether to access raw when there are keys in var_names."
    )
    
    text_kwargs: Dict[str, Any] = Field(
        default={},
        description="Keyword arguments for text()."
    )
    
    label_distance: float = Field(
        default=1.25,
        description="Distance at which the lineage labels will be drawn."
    )
    
    label_rot: Union[Literal['default', 'best'], float] = Field(
        default='best',
        description="How to rotate the labels. 'best' rotates for readability, 'default' uses matplotlib's default, float rotates by degrees."
    )
    
    show_edges: bool = Field(
        default=True,
        description="Whether to show the edges surrounding the simplex."
    )
    
    key_added: Optional[str] = Field(
        default=None,
        description="Key in obsm where to add the circular embedding. If None, it will be set to 'X_fate_simplex_{fwd,bwd}'."
    )
    
    figsize: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Size of the figure."
    )
    
    dpi: Optional[int] = Field(
        default=None,
        description="Dots per inch."
    )
    
    save: Optional[Union[str, bool]] = Field(
        default=None,
        description="Filename where to save the plot."
    )
    
    # Common plotting parameters
    color: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Key for annotations of observations/cells or variables/genes."
    )
    
    alpha: Optional[float] = Field(
        default=None,
        description="Alpha value for the scatter plot."
    )
    
    size: Optional[Union[float, int]] = Field(
        default=None,
        description="Point size for the scatter plot."
    )
    
    legend_loc: Optional[str] = Field(
        default="right",
        description="Location of legend, either 'on data', 'right margin' or valid keywords for matplotlib.legend."
    )
    
    title: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Title for the plot."
    )