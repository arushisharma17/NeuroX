from .algorithms.clustering import AgglomerativeClusteringPipeline, KMeansClusteringPipeline, LeadersClusteringPipeline
from .evaluation import alignment

__all__ = [
    "AgglomerativeClusteringPipeline",
    "KMeansClusteringPipeline",
    "LeadersClusteringPipeline",
    "alignment"
]

__version__ = "0.1.0"  # test to see pip working 