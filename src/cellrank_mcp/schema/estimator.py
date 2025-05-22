from pydantic import (
    Field,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator, BaseModel
)
from typing import Optional, List, Dict, Union, Literal, Tuple, Any, Sequence


class GPCCAFitModel(BaseModel):
    """
    Input schema for CellRank's GPCCA.fit method which prepares the estimator for terminal states prediction.
    """
    kernel: Literal['pseudotime', 'cytotrace', 'velocity', 'connectivity', 'realtime'] = Field(
        description="Type of kernel to use."
    )
    
    n_states: Optional[Union[int, List[int]]] = Field(
        default=None,
        description="Number of macrostates to compute. If a list, use the minChi criterion. If None, use the eigengap heuristic."
    )
    
    n_cells: Optional[int] = Field(
        default=30,
        description="Number of most likely cells from each macrostate to select."
    )
    
    cluster_key: Optional[str] = Field(
        default=None,
        description="If a key to cluster labels is given, names and colors of the states will be associated with the clusters."
    )
    
    # Parameters for compute_schur method
    n_components: int = Field(
        default=20,
        description="Number of Schur vectors to compute."
    )
    
    initial_distribution: Optional[Any] = Field(
        default=None,
        description="Input distribution over all cells. If None, uniform distribution is used."
    )
    
    method: Literal['krylov', 'brandts'] = Field(
        default='krylov',
        description="Method for calculating the Schur vectors. 'krylov' is an iterative procedure for large, sparse matrices. 'brandts' is full sorted Schur decomposition of a dense matrix."
    )
    
    which: Literal['LR', 'LM'] = Field(
        default='LR',
        description="How to sort the eigenvalues. 'LR' - the largest real part. 'LM' - the largest magnitude."
    )
    
    alpha: float = Field(
        default=1.0,
        description="Used to compute the eigengap. alpha is the weight given to the deviation of an eigenvalue from one."
    )


class GPCCAPredictInitialStatesModel(BaseModel):
    """
    Input schema for CellRank's GPCCA.predict_initial_states method which computes initial states from macrostates.
    """
    kernel: Literal['pseudotime', 'cytotrace', 'velocity', 'connectivity', 'realtime'] = Field(
        ...,
        description="Type of kernel to use."
    )
    n_states: int = Field(
        default=1,
        description="Number of initial states to compute."
    )
    
    n_cells: int = Field(
        default=30,
        description="Number of most likely cells from each macrostate to select."
    )
    
    allow_overlap: bool = Field(
        default=False,
        description="Whether to allow overlapping names between initial and terminal states."
    )


class GPCCAPredictTerminalStatesModel(BaseModel):
    """
    Input schema for CellRank's GPCCA.predict_terminal_states method which automatically selects terminal states from macrostates.
    """
    kernel: Literal['pseudotime', 'cytotrace', 'velocity', 'connectivity', 'realtime'] = Field(
        description="Type of kernel to use."
    )
    method: Literal['stability', 'top_n', 'eigengap', 'eigengap_coarse'] = Field(
        default='stability',
        description="How to select the terminal states. 'eigengap' - select based on eigengap of transition_matrix. "
                   "'eigengap_coarse' - select based on eigengap of diagonal of coarse_T. "
                   "'top_n' - select top n_states based on probability of diagonal of coarse_T. "
                   "'stability' - select states with stability >= stability_threshold."
    )
    
    n_cells: int = Field(
        default=30,
        description="Number of most likely cells from each macrostate to select."
    )
    
    alpha: float = Field(
        default=1,
        description="Weight given to the deviation of an eigenvalue from one. Only used when method = 'eigengap' or method = 'eigengap_coarse'."
    )
    
    stability_threshold: float = Field(
        default=0.96,
        description="Threshold used when method = 'stability'."
    )
    
    n_states: Optional[int] = Field(
        default=None,
        description="Number of states used when method = 'top_n'."
    )
    
    allow_overlap: bool = Field(
        default=False,
        description="Whether to allow overlapping names between initial and terminal states."
    )


class GPCCAComputeFateProbabilitiesModel(BaseModel):
    """
    Input schema for CellRank's GPCCA.compute_fate_probabilities method which calculates the probability
    of each cell being absorbed in any of the terminal states.
    """
    kernel: Literal['pseudotime', 'cytotrace', 'velocity', 'connectivity', 'realtime'] = Field(
        description="Type of kernel to use."
    )
    keys: Optional[List[str]] = Field(
        default=None,
        description="Terminal states for which to compute the fate probabilities. If None, use all states defined in terminal_states."
    )
    
    solver: Literal['direct', 'gmres', 'lgmres', 'bicgstab', 'gcrotmk'] = Field(
        default='gmres',
        description="Solver to use for the linear problem. Options are 'direct', 'gmres', 'lgmres', 'bicgstab' or 'gcrotmk' when use_petsc = False."
    )
    
    use_petsc: bool = Field(
        default=True,
        description="Whether to use solvers from petsc4py or scipy. Recommended for large problems."
    )
    
    n_jobs: Optional[int] = Field(
        default=None,
        description="Number of parallel jobs to use when using an iterative solver."
    )
    
    backend: Literal['loky', 'multiprocessing', 'threading'] = Field(
        default='loky',
        description="Which backend to use for multiprocessing."
    )
    
    tol: float = Field(
        default=1e-06,
        description="Convergence tolerance for the iterative solver."
    )
    
    preconditioner: Optional[str] = Field(
        default=None,
        description="Preconditioner to use, only available when use_petsc = True. We recommend the 'ilu' preconditioner for badly conditioned problems."
    )
    
