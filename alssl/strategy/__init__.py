from .alssl.neighbours import NeighboursStrategy
from .alssl.neighbours_coreset import FFStrongStrategy
from .alssl.randomized_entropy import RandEntropyStrategy
from .alssl.umaplike import UMAPLikeStrategy
from .badge import BADGEStrategy
from .bait import BAITStrategy
from .cal import CALStrategy
from .cdal import CDALStrategy
from .coreset import CoresetStrategy
from .entropy import EntropyStrategy
# from .alssl.neighbours_path import NeighboursPathStrategy
from .golden_rule import GRStrategy
from .random import RandomStrategy

locals = locals()
strategies = {key: locals[key] for key in locals if key.endswith('Strategy')}
