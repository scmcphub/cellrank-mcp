from pydantic import (
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,BaseModel
)
from typing import Optional, List, Dict, Union, Literal, Any, Tuple


class PseudotimeKernelModel(BaseModel):
    """Input schema for CellRank's PseudotimeKernel which computes directed transition probabilities based on a k-NN graph and pseudotime."""
    
    time_key: str = Field(
        description="Key in AnnData.obs where the pseudotime is stored."
    )
    backward: bool = Field(
        default=False,
        description="Direction of the process. If True, the pseudotime will be set to max(pseudotime) - pseudotime."
    )


class CytoTRACEKernelModel(BaseModel):
    """Input schema for CellRank's CytoTRACEKernel which computes directed transition probabilities using the CytoTRACE score."""
    
    backward: bool = Field(
        default=False,
        description="Direction of the process. If True, the CytoTRACE score will be inverted."
    )


class VelocityKernelModel(BaseModel):
    """Input schema for CellRank's VelocityKernel which computes a transition matrix based on RNA velocity."""
    
    backward: bool = Field(
        default=False,
        description="Direction of the process. If True, velocity vectors will be reversed."
    )
    
    attr: Optional[Literal['layers', 'obsm']] = Field(
        default='layers',
        description="Attribute of AnnData to read from."
    )
    
    xkey: Optional[str] = Field(
        default='Ms',
        description="Key in AnnData.layers or AnnData.obsm where expected gene expression counts are stored."
    )
    
    vkey: Optional[str] = Field(
        default='velocity',
        description="Key in AnnData.layers or AnnData.obsm where velocities are stored."
    )
    
    gene_subset: Optional[List[str]] = Field(
        default=None,
        description="List of genes to be used to compute transition probabilities. If not specified, genes from adata.var['{vkey}_genes'] are used."
    )


class ConnectivityKernelModel(BaseModel):
    """Input schema for CellRank's ConnectivityKernel which computes transition probabilities based on similarities among cells."""
    
    conn_key: str = Field(
        default='connectivities',
        description="Key in AnnData.obsp where connectivity matrix describing cell-cell similarity is stored."
    )
    
    check_connectivity: bool = Field(
        default=False,
        description="Check whether the underlying kNN graph is connected."
    )


class RealTimeKernelModel(BaseModel):
    """Input schema for creating a RealTimeKernel"""
    
    time_key: str = Field(
        description="Key in AnnData.obs containing the experimental time."
    )
    # Add couplings field
    couplings: Optional[Dict[Tuple[Any, Any], Any]] = Field(
        default=None,
        description="Pre-computed transport couplings. The keys should correspond to a tuple of categories from the time. If None, the keys will be constructed using the policy and the compute_coupling() method must be overriden."
    )
    policy: Literal['sequential', 'triu'] = Field(
        default='sequential',
        description="How to construct keys from time: 'sequential' for [(t1,t2),(t2,t3),...], 'triu' for [(t1,t2),(t1,t3),...,(t2,t3),...]."
    )
    


class KernelModel(BaseModel):
    """Unified input schema for creating various CellRank Kernels."""
    
    kernel: Literal['pseudotime', 'cytotrace', 'velocity', 'connectivity', 'realtime'] = Field(
        description="Type of kernel to use."
    )
    
    # PseudotimeKernel and RealTimeKernel specific fields
    time_key: Optional[str] = Field(
        default=None,
        description="[pseudotime, realtime] Key in AnnData.obs where the pseudotime or experimental time is stored."
    )
    
    # Shared fields across multiple kernels
    backward: Optional[bool] = Field(
        default=False,
        description="[pseudotime, cytotrace, velocity] Direction of the process. If True, the direction will be reversed."
    )
    
    # VelocityKernel specific fields
    attr: Optional[Literal['layers', 'obsm']] = Field(
        default='layers',
        description="[velocity] Attribute of AnnData to read from."
    )
    
    xkey: Optional[str] = Field(
        default='Ms',
        description="[velocity] Key in AnnData.layers or AnnData.obsm where expected gene expression counts are stored."
    )
    
    vkey: Optional[str] = Field(
        default='velocity',
        description="[velocity] Key in AnnData.layers or AnnData.obsm where velocities are stored."
    )
    
    gene_subset: Optional[List[str]] = Field(
        default=None,
        description="[velocity] List of genes to be used to compute transition probabilities. If not specified, genes from adata.var['{vkey}_genes'] are used."
    )
    
    # ConnectivityKernel specific fields
    conn_key: Optional[str] = Field(
        default='connectivities',
        description="[connectivity] Key in AnnData.obsp where connectivity matrix describing cell-cell similarity is stored."
    )
    
    check_connectivity: Optional[bool] = Field(
        default=False,
        description="[connectivity] Check whether the underlying kNN graph is connected."
    )
    
    # RealTimeKernel specific fields
    couplings: Optional[Dict[Tuple[Any, Any], Any]] = Field(
        default=None,
        description="[realtime] Pre-computed transport couplings. The keys should correspond to a tuple of categories from the time. If None, the keys will be constructed using the policy and the compute_coupling() method must be overriden."
    )
    
    policy: Optional[Literal['sequential', 'triu']] = Field(
        default='sequential',
        description="[realtime] How to construct keys from time: 'sequential' for [(t1,t2),(t2,t3),...], 'triu' for [(t1,t2),(t1,t3),...,(t2,t3),...]."
    )



class VelocityComputeTransitionMatrixModel(BaseModel):
    """Input schema for computing transition matrix based on velocity directions on the local manifold in CellRank."""
    
    model: Literal['deterministic', 'stochastic', 'monte_carlo'] = Field(
        default='deterministic',
        description="How to compute transition probabilities. 'deterministic' doesn't propagate uncertainty, 'monte_carlo' uses random sampling, 'stochastic' uses second order approximation (requires jax)."
    )
    
    backward_mode: Literal['transpose', 'negate'] = Field(
        default='transpose',
        description="Only matters if kernel initialized with backward=True. 'transpose' computes transitions from neighboring cells to cell i, 'negate' negates the velocity vector."
    )
    
    similarity: Literal['correlation', 'cosine', 'dot_product'] = Field(
        default='correlation',
        description="Similarity measure between cells as described in Li et al., 2021."
    )
    
    softmax_scale: Optional[float] = Field(
        default=None,
        description="Scaling parameter for the softmax. If None, estimated using 1 / median(correlations) to counter high-dimensional orthogonality."
    )
    
    n_samples: int = Field(
        default=1000,
        description="Number of samples when model='monte_carlo'."
    )
    
    seed: Optional[int] = Field(
        default=None,
        description="Random seed when model='monte_carlo'."
    )
    
    n_jobs: Optional[int] = Field(
        default=None,
        description="Number of parallel jobs. If -1, use all available cores. If None or 1, execution is sequential."
    )
    
    backend: Literal['loky', 'multiprocessing', 'threading'] = Field(
        default='loky',
        description="Which backend to use for parallelization."
    )


class ConnectivityComputeTransitionMatrixModel(BaseModel):
    """Input schema for computing transition matrix based on transcriptomic similarity in CellRank's ConnectivityKernel."""
    
    density_normalize: bool = Field(
        default=True,
        description="Whether to use the underlying kNN graph for density normalization."
    )


class PseudotimeComputeTransitionMatrixModel(BaseModel):
    """Input schema for computing transition matrix based on k-NN graph and pseudotemporal ordering in CellRank's PseudotimeKernel."""
    
    threshold_scheme: Literal['soft', 'hard'] = Field(
        default='hard',
        description="Which method to use when biasing the graph. 'hard' removes edges against pseudotime direction (Palantir), 'soft' down-weights them (VIA)."
    )
    
    frac_to_keep: float = Field(
        default=0.3,
        description="Fraction of closest neighbors to keep regardless of pseudotime. Only used with 'hard' scheme. Must be in [0, 1]."
    )
    
    b: float = Field(
        default=10.0,
        description="Growth rate of generalized logistic function. Only used with 'soft' scheme."
    )
    
    nu: float = Field(
        default=0.5,
        description="Affects near which asymptote maximum growth occurs. Only used with 'soft' scheme."
    )
    
    check_irreducibility: bool = Field(
        default=False,
        description="Whether to check for irreducibility of the final transition matrix."
    )
    
    n_jobs: Optional[int] = Field(
        default=None,
        description="Number of parallel jobs. If -1, use all available cores. If None or 1, execution is sequential."
    )
    
    backend: Literal['loky', 'multiprocessing', 'threading'] = Field(
        default='loky',
        description="Which backend to use for parallelization."
    )
    

class CytoTRACEComputeTransitionMatrixModel(BaseModel):
    """Input schema for computing transition matrix based on k-NN graph and CytoTRACE scores in CellRank's CytoTRACEKernel."""
    
    threshold_scheme: Literal['soft', 'hard'] = Field(
        default='hard',
        description="Which method to use when biasing the graph. 'hard' removes edges against CytoTRACE direction (Palantir), 'soft' down-weights them (VIA)."
    )
    
    frac_to_keep: float = Field(
        default=0.3,
        description="Fraction of closest neighbors to keep regardless of CytoTRACE score. Only used with 'hard' scheme. Must be in [0, 1]."
    )
    
    b: float = Field(
        default=10.0,
        description="Growth rate of generalized logistic function. Only used with 'soft' scheme."
    )
    
    nu: float = Field(
        default=0.5,
        description="Affects near which asymptote maximum growth occurs. Only used with 'soft' scheme."
    )
    
    check_irreducibility: bool = Field(
        default=False,
        description="Whether to check for irreducibility of the final transition matrix."
    )
    
    n_jobs: Optional[int] = Field(
        default=None,
        description="Number of parallel jobs. If -1, use all available cores. If None or 1, execution is sequential."
    )
    
    backend: Literal['loky', 'multiprocessing', 'threading'] = Field(
        default='loky',
        description="Which backend to use for parallelization."
    )
    

class RealTimeComputeTransitionMatrixModel(BaseModel):
    """Input schema for computing transition matrix from optimal transport couplings in CellRank's RealTimeKernel."""
    
    threshold: Union[int, float, Literal['auto', 'auto_local'], None] = Field(
        default='auto',
        description="How to remove small non-zero values from the transition matrix. 'auto' finds maximum threshold that keeps at least one non-zero per row, 'auto_local' does this per transport, float is percentage of non-zeros to remove."
    )
    
    self_transitions: Union[Literal['uniform', 'diagonal', 'connectivities', 'all'], List[Any]] = Field(
        default='connectivities',
        description="How to define transitions within blocks corresponding to same time point. 'uniform' for row-normalized matrix, 'diagonal' for identity, 'connectivities' for ConnectivityKernel transitions, 'all' or sequence for specific blocks."
    )
    
    conn_weight: Optional[float] = Field(
        default=None,
        description="Weight of connectivities' self transitions. Only used when self_transitions='all' or a sequence of source keys is passed."
    )
    
    conn_kwargs: Dict[str, Any] = Field(
        default={},
        description="Keyword arguments for neighbors() or compute_transition_matrix() when using self_transitions='connectivities'."
    )
    

class ComputeTransitionMatrixModel(BaseModel):
    """Unified input schema for computing transition matrices across different CellRank kernel types."""
    kernel: Literal['pseudotime', 'cytotrace', 'velocity', 'connectivity', 'realtime'] = Field(
        description="Type of kernel to use."
    )
    
    # Common parameters across multiple kernels
    n_jobs: Optional[int] = Field(
        default=None,
        description="[velocity, pseudotime, cytotrace] Number of parallel jobs. If -1, use all available cores. If None or 1, execution is sequential."
    )
    
    backend: Literal['loky', 'multiprocessing', 'threading'] = Field(
        default='loky',
        description="[velocity, pseudotime, cytotrace] Which backend to use for parallelization."
    )
    
    check_irreducibility: bool = Field(
        default=False,
        description="[pseudotime, cytotrace] Whether to check for irreducibility of the final transition matrix."
    )
    
    # VelocityKernel specific parameters
    model: Optional[Literal['deterministic', 'stochastic', 'monte_carlo']] = Field(
        default=None,
        description="[velocity] How to compute transition probabilities. 'deterministic' doesn't propagate uncertainty, 'monte_carlo' uses random sampling, 'stochastic' uses second order approximation (requires jax)."
    )
    
    backward_mode: Optional[Literal['transpose', 'negate']] = Field(
        default=None,
        description="[velocity] Only matters if kernel initialized with backward=True. 'transpose' computes transitions from neighboring cells to cell i, 'negate' negates the velocity vector."
    )
    
    similarity: Optional[Literal['correlation', 'cosine', 'dot_product']] = Field(
        default=None,
        description="[velocity] Similarity measure between cells as described in Li et al., 2021."
    )
    
    softmax_scale: Optional[float] = Field(
        default=None,
        description="[velocity] Scaling parameter for the softmax. If None, estimated using 1 / median(correlations) to counter high-dimensional orthogonality."
    )
    
    n_samples: Optional[int] = Field(
        default=None,
        description="[velocity] Number of samples when model='monte_carlo'."
    )
    
    # ConnectivityKernel specific parameters
    density_normalize: Optional[bool] = Field(
        default=None,
        description="[connectivity] Whether to use the underlying kNN graph for density normalization."
    )
    
    # PseudotimeKernel and CytoTRACEKernel shared parameters
    threshold_scheme: Optional[Literal['soft', 'hard']] = Field(
        default=None,
        description="[pseudotime, cytotrace] Which method to use when biasing the graph. 'hard' removes edges against direction, 'soft' down-weights them."
    )
    
    frac_to_keep: Optional[float] = Field(
        default=None,
        description="[pseudotime, cytotrace] Fraction of closest neighbors to keep regardless of ordering. Only used with 'hard' scheme. Must be in [0, 1]."
    )
    
    b: Optional[float] = Field(
        default=None,
        description="[pseudotime, cytotrace] Growth rate of generalized logistic function. Only used with 'soft' scheme."
    )
    
    nu: Optional[float] = Field(
        default=None,
        description="[pseudotime, cytotrace] Affects near which asymptote maximum growth occurs. Only used with 'soft' scheme."
    )
    
    # RealTimeKernel specific parameters
    threshold: Optional[Union[int, float, Literal['auto', 'auto_local'], None]] = Field(
        default=None,
        description="[realtime] How to remove small non-zero values from the transition matrix. 'auto' finds maximum threshold that keeps at least one non-zero per row."
    )
    
    self_transitions: Optional[Union[Literal['uniform', 'diagonal', 'connectivities', 'all'], List[Any]]] = Field(
        default=None,
        description="[realtime] How to define transitions within blocks corresponding to same time point."
    )
    
    conn_weight: Optional[float] = Field(
        default=None,
        description="[realtime] Weight of connectivities' self transitions. Only used when self_transitions='all' or a sequence of source keys is passed."
    )
    
    conn_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="[realtime] Keyword arguments for neighbors() or compute_transition_matrix() when using self_transitions='connectivities'."
    )
