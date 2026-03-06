"""Create heatmaps from NetworkX graphs."""

from importlib.metadata import version

__version__ = version('heatmapx')
__all__ = ['temperature_graph']

from heatmapx._thermograph import temperature_graph
