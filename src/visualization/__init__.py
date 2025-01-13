"""
Environmental data visualization package.
Provides tools for generating visualizations of environmental data.
"""

from .generator import VisualizationGenerator
from .utils import EnvironmentalData
from .color_schemes import COLOR_SCHEMES

__all__ = ['VisualizationGenerator', 'EnvironmentalData', 'COLOR_SCHEMES'] 