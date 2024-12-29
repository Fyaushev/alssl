from .kmeans import KMeansColdStart
from .random import RandomColdStart
from .turtle.turtle import TurtleColdStart

locals = locals()
coldstarts = {key: locals[key] for key in locals if key.endswith('ColdStart')}
