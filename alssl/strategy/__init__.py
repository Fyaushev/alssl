from .alssl.neighbours import NeighboursStrategy
from .alssl.neighbours_coreset import FFStrongStrategy
from .alssl.neighbours_max_coreset import FFNNMaxStrategy
from .alssl.randomized_entropy import RandEntropyStrategy
from .alssl.umaplike import UMAPLikeStrategy
from .badge import BADGEStrategy
from .bait import BAITStrategy
from .cal import CALStrategy
from .cdal import CDALStrategy
from .coreset import CoresetStrategy
from .dcom import DCoMStrategy
from .embclust import EmbClustStrategy
from .entropy import EntropyStrategy
# from .alssl.neighbours_path import NeighboursPathStrategy
from .golden_rule import GRStrategy
from .kmeans import KMeansStrategy
from .neighbors import NNStrategy
from .prob_cover import ProbCoverStrategy
from .random import RandomStrategy
from .typiclust import TypiClustStrategy
from .typistable import TypiStableStrategy

locals = locals()
strategies = {key: locals[key] for key in locals if key.endswith('Strategy')}
