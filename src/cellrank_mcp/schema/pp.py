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
