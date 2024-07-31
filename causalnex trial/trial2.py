import warnings
from causalnex.structure import StructureModel
warnings.filterwarnings("ignore")  # silence warnings

sm = StructureModel()
sm.add_edges_from([
    ('health', 'absences'),
    ('health', 'G1')
])
sm.edges

from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

viz = plot_structure(
    sm,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)

viz.draw('trial2.png')