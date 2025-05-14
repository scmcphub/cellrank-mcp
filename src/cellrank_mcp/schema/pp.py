from pydantic import (
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
    BaseModel
)
from typing import Optional, List, Dict, Union, Literal, Tuple, Any


class FilterAndNormalizeModel(BaseModel):
    """Input schema for filter_and_normalize preprocessing function."""
    
    min_counts: Optional[int] = Field(
        default=None,
        description="Minimum number of counts required for a gene to pass filtering (spliced)."
    )
    min_counts_u: Optional[int] = Field(
        default=None,
        description="Minimum number of counts required for a gene to pass filtering (unspliced)."
    )
    min_cells: Optional[int] = Field(
        default=None,
        description="Minimum number of cells expressed required to pass filtering (spliced)."
    )
    min_cells_u: Optional[int] = Field(
        default=None,
        description="Minimum number of cells expressed required to pass filtering (unspliced)."
    )
    min_shared_counts: Optional[int] = Field(
        default=None,
        description="Minimum number of counts (both unspliced and spliced) required for a gene."
    )
    min_shared_cells: Optional[int] = Field(
        default=None,
        description="Minimum number of cells required to be expressed (both unspliced and spliced)."
    )
    n_top_genes: Optional[int] = Field(
        default=None,
        description="Number of genes to keep."
    )
    retain_genes: Optional[List[str]] = Field(
        default=None,
        description="List of gene names to be retained independent of thresholds."
    )
    subset_highly_variable: bool = Field(
        default=True,
        description="Whether to subset highly variable genes or to store in .var['highly_variable']."
    )
    
    flavor: Literal["seurat", "cell_ranger", "svr"] = Field(
        default="seurat",
        description="Choose the flavor for computing normalized dispersion. If choosing 'seurat', this expects non-logarithmized data."
    )
    
    layers_normalize: Optional[List[str]] = Field(
        default=None,
        description="List of layers to be normalized. If set to None, the layers {'X', 'spliced', 'unspliced'} are considered for normalization."
    )
    target_sum: Optional[float] = Field(
            default=None,
            description="If None, after normalization, each observation (cell) has a total count equal to the median of total counts for observations (cells) before normalization."
        )


class TSNEModel(BaseModel):
    """Input schema for the t-SNE dimensionality reduction tool."""
    n_pcs: Optional[int] = Field(
        default=None,
        description="Number of PCs to use. If None, automatically determined.",
        ge=0
    )
    use_rep: Optional[str] = Field(
        default=None,
        description="Key for .obsm to use as representation."
    )
    perplexity: Union[float, int] = Field(
        default=30,
        description="Related to number of nearest neighbors, typically between 5-50.",
        gt=0
    )
    early_exaggeration: Union[float, int] = Field(
        default=12,
        description="Controls space between natural clusters in embedded space.",
        gt=0
    )
    learning_rate: Union[float, int] = Field(
        default=1000,
        description="Learning rate for optimization, typically between 100-1000.",
        gt=0
    )
    random_state: int = Field(
        default=0,
        description="Random seed for reproducibility."
    )
    use_fast_tsne: bool = Field(
        default=False,
        description="Whether to use Multicore-tSNE implementation."
    )
    n_jobs: Optional[int] = Field(
        default=None,
        description="Number of jobs for parallel computation.",
        gt=0
    )
    metric: str = Field(
        default='euclidean',
        description="Distance metric to use."
    )
    
    @field_validator('n_pcs', 'perplexity', 'early_exaggeration', 
                   'learning_rate', 'n_jobs')
    def validate_positive_numbers(cls, v: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
        """Validate positive numbers where applicable"""
        if v is not None and v <= 0:
            raise ValueError("must be a positive number")
        return v
    
    @field_validator('metric')
    def validate_metric(cls, v: str) -> str:
        """Validate distance metric is supported"""
        valid_metrics = ['euclidean', 'cosine', 'manhattan', 'l1', 'l2']
        if v.lower() not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
        return v.lower()


class UMAPModel(BaseModel):
    """Input schema for the UMAP dimensionality reduction tool."""
    
    min_dist: float = Field(
        default=0.5,
        description="Minimum distance between embedded points.",
        gt=0
    )
    
    spread: float = Field(
        default=1.0,
        description="Scale of embedded points.",
        gt=0
    )
    
    n_components: int = Field(
        default=2,
        description="Number of dimensions of the embedding.",
        gt=0
    )
    
    maxiter: Optional[int] = Field(
        default=None,
        description="Number of iterations (epochs) of the optimization.",
        gt=0
    )
    
    alpha: float = Field(
        default=1.0,
        description="Initial learning rate for the embedding optimization.",
        gt=0
    )
    
    gamma: float = Field(
        default=1.0,
        description="Weighting applied to negative samples.",
        gt=0
    )
    negative_sample_rate: int = Field(
        default=5,
        description="Number of negative samples per positive sample.",
        gt=0
    )
    init_pos: str = Field(
        default='spectral',
        description="How to initialize the low dimensional embedding.",
    )
    random_state: int = Field(
        default=0,
        description="Random seed for reproducibility."
    )
    a: Optional[float] = Field(
        default=None,
        description="Parameter controlling the embedding.",
        gt=0
    )
    b: Optional[float] = Field(
        default=None,
        description="Parameter controlling the embedding.",
        gt=0
    )
    method: str = Field(
        default='umap',
        description="Implementation to use ('umap' or 'rapids')."
    )
    neighbors_key: Optional[str] = Field(
        default=None,
        description="Key for neighbors settings in .uns."
    )
    
    @field_validator('min_dist', 'spread', 'n_components', 'maxiter', 
                   'alpha', 'gamma', 'negative_sample_rate', 'a', 'b')
    def validate_positive_numbers(cls, v: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
        """Validate positive numbers where applicable"""
        if v is not None and v <= 0:
            raise ValueError("must be a positive number")
        return v
    
    @field_validator('method')
    def validate_method(cls, v: str) -> str:
        """Validate implementation method is supported"""
        if v.lower() not in ['umap', 'rapids']:
            raise ValueError("method must be either 'umap' or 'rapids'")
        return v.lower()


class DiffMapModel(BaseModel):
    """Input schema for the Diffusion Maps dimensionality reduction tool."""
    
    n_comps: int = Field(
        default=15,
        description="The number of dimensions of the representation.",
        gt=0
    )
    neighbors_key: Optional[str] = Field(
        default=None,
        description=(
            "If not specified, diffmap looks .uns['neighbors'] for neighbors settings "
            "and .obsp['connectivities'], .obsp['distances'] for connectivities and "
            "distances respectively. If specified, diffmap looks .uns[neighbors_key] for "
            "neighbors settings and uses the corresponding connectivities and distances."
        )
    )
    random_state: int = Field(
        default=0,
        description="Random seed for reproducibility."
    )
    
    @field_validator('n_comps')
    def validate_positive_integers(cls, v: int) -> int:
        """Validate positive integers"""
        if v <= 0:
            raise ValueError("n_comps must be a positive integer")
        return v



class LeidenModel(BaseModel):
    """Input schema for the Leiden clustering algorithm."""
    
    resolution: float = Field(
        default=1.0,
        description="A parameter value controlling the coarseness of the clustering. Higher values lead to more clusters."
    )
    
    random_state: int = Field(
        default=0,
        description="Change the initialization of the optimization."
    )
    
    key_added: str = Field(
        default='leiden',
        description="`adata.obs` key under which to add the cluster labels."
    )
    
    directed: Optional[bool] = Field(
        default=None,
        description="Whether to treat the graph as directed or undirected."
    )
    
    use_weights: bool = Field(
        default=True,
        description="If `True`, edge weights from the graph are used in the computation (placing more emphasis on stronger edges)."
    )
    
    n_iterations: int = Field(
        default=-1,
        description="How many iterations of the Leiden clustering algorithm to perform. -1 runs until optimal clustering."
    )
    
    neighbors_key: Optional[str] = Field(
        default=None,
        description="Use neighbors connectivities as adjacency. If specified, leiden looks .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities."
    )
    
    obsp: Optional[str] = Field(
        default=None,
        description="Use .obsp[obsp] as adjacency. You can't specify both `obsp` and `neighbors_key` at the same time."
    )
    
    flavor: Literal['leidenalg', 'igraph'] = Field(
        default='igraph',
        description="Which package's implementation to use."
    )
    
    clustering_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Any further arguments to pass to the clustering algorithm."
    )
    
    @field_validator('resolution')
    def validate_resolution(cls, v: float) -> float:
        """Validate resolution is positive"""
        if v <= 0:
            raise ValueError("resolution must be a positive number")
        return v
    
    @field_validator('obsp', 'neighbors_key')
    def validate_graph_source(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate that obsp and neighbors_key are not both specified"""
        values = info.data
        if v is not None and 'obsp' in values and 'neighbors_key' in values:
            if values['obsp'] is not None and values['neighbors_key'] is not None:
                raise ValueError("Cannot specify both obsp and neighbors_key")
        return v
    
    @field_validator('flavor')
    def validate_flavor(cls, v: str) -> str:
        """Validate flavor is supported"""
        if v not in ['leidenalg', 'igraph']:
            raise ValueError("flavor must be either 'leidenalg' or 'igraph'")
        return v



class DPTModel(BaseModel):
    """Input schema for the Diffusion Pseudotime (DPT) tool."""
    
    n_dcs: int = Field(
        default=10,
        description="The number of diffusion components to use.",
        gt=0
    )
    n_branchings: int = Field(
        default=0,
        description="Number of branchings to detect.",
        ge=0
    )
    min_group_size: float = Field(
        default=0.01,
        description="During recursive splitting of branches, do not consider groups that contain less than min_group_size data points. If a float, refers to a fraction of the total number of data points.",
        gt=0,
        le=1.0
    )
    allow_kendall_tau_shift: bool = Field(
        default=True,
        description="If a very small branch is detected upon splitting, shift away from maximum correlation in Kendall tau criterion to stabilize the splitting."
    )
    neighbors_key: Optional[str] = Field(
        default=None,
        description="If specified, dpt looks .uns[neighbors_key] for neighbors settings and uses the corresponding connectivities and distances."
    )
    
    @field_validator('n_dcs')
    def validate_n_dcs(cls, v: int) -> int:
        """Validate n_dcs is positive"""
        if v <= 0:
            raise ValueError("n_dcs must be a positive integer")
        return v
    
    @field_validator('n_branchings')
    def validate_n_branchings(cls, v: int) -> int:
        """Validate n_branchings is non-negative"""
        if v < 0:
            raise ValueError("n_branchings must be a non-negative integer")
        return v
    
    @field_validator('min_group_size')
    def validate_min_group_size(cls, v: float) -> float:
        """Validate min_group_size is between 0 and 1"""
        if v <= 0 or v > 1:
            raise ValueError("min_group_size must be between 0 and 1")
        return v



class NeighborsModel(BaseModel):
    """Input schema for the neighbors graph construction tool."""
    
    n_neighbors: int = Field(
        default=15,
        description="Size of local neighborhood used for manifold approximation.",
        gt=1,
        le=100
    )
    
    n_pcs: Optional[int] = Field(
        default=None,
        description="Number of PCs to use. If None, automatically determined.",
        ge=0
    )
    
    use_rep: Optional[str] = Field(
        default=None,
        description="Key for .obsm to use as representation."
    )
    
    knn: bool = Field(
        default=True,
        description="Whether to use hard threshold for neighbor restriction."
    )
    
    method: Literal['umap', 'gauss'] = Field(
        default='umap',
        description="Method for computing connectivities ('umap' or 'gauss')."
    )
    
    transformer: Optional[str] = Field(
        default=None,
        description="Approximate kNN search implementation ('pynndescent' or 'rapids')."
    )
    
    metric: str = Field(
        default='euclidean',
        description="Distance metric to use."
    )
    
    metric_kwds: Dict[str, Any] = Field(
        default_factory=dict,
        description="Options for the distance metric."
    )
    
    random_state: int = Field(
        default=0,
        description="Random seed for reproducibility."
    )
    
    key_added: Optional[str] = Field(
        default=None,
        description="Key prefix for storing neighbor results."
    )
    
    @field_validator('n_neighbors', 'n_pcs')
    def validate_positive_integers(cls, v: Optional[int]) -> Optional[int]:
        """Validate positive integers where applicable"""
        if v is not None and v <= 0:
            raise ValueError("must be a positive integer")
        return v
    
    @field_validator('method')
    def validate_method(cls, v: str) -> str:
        """Validate method is supported"""
        if v not in ['umap', 'gauss']:
            raise ValueError("method must be either 'umap' or 'gauss'")
        return v
    
    @field_validator('transformer')
    def validate_transformer(cls, v: Optional[str]) -> Optional[str]:
        """Validate transformer option is supported"""
        if v is not None and v not in ['pynndescent', 'rapids']:
            raise ValueError("transformer must be either 'pynndescent' or 'rapids'")
        return v


class PCAModel(BaseModel):
    """Input schema for the PCA preprocessing tool."""
    
    n_comps: Optional[int] = Field(
        default=None,
        description="Number of principal components to compute. Defaults to 50 or 1 - minimum dimension size.",
        gt=0
    )
    
    layer: Optional[str] = Field(
        default=None,
        description="If provided, which element of layers to use for PCA."
    )
    
    zero_center: Optional[bool] = Field(
        default=True,
        description="If True, compute standard PCA from covariance matrix."
    )
    
    svd_solver: Optional[Literal["arpack", "randomized", "auto", "lobpcg", "tsqr"]] = Field(
        default=None,
        description="SVD solver to use."
    )
    mask_var: Optional[Union[str, bool]] = Field(
        default=None,
        description="Boolean mask or string referring to var column for subsetting genes."
    )
    dtype: str = Field(
        default="float32",
        description="Numpy data type string for the result."
    )
    chunked: bool = Field(
        default=False,
        description="If True, perform an incremental PCA on segments."
    )
    
    chunk_size: Optional[int] = Field(
        default=None,
        description="Number of observations to include in each chunk.",
        gt=0
    )
    
    @field_validator('n_comps', 'chunk_size')
    def validate_positive_integers(cls, v: Optional[int]) -> Optional[int]:
        """Validate positive integers"""
        if v is not None and v <= 0:
            raise ValueError("must be a positive integer")
        return v
    
    @field_validator('dtype')
    def validate_dtype(cls, v: str) -> str:
        """Validate numpy dtype"""
        if v not in ["float32", "float64"]:
            raise ValueError("dtype must be either 'float32' or 'float64'")
        return v
