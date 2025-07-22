"""Color schemes for visualizations."""

from typing import Dict, List, Tuple
from enum import Enum


class ColorScheme(str, Enum):
    """Available color schemes."""
    REDBLUE = "redblue"
    CYAN = "cyan"
    BLUE = "blue"
    RED = "red"
    BASE = "base"
    YELLOW = "yellow"
    PURPLE = "purple"
    GREEN = "green"


# Color scheme definitions (hex colors)
COLOR_SCHEMES: Dict[ColorScheme, List[str]] = {
    ColorScheme.REDBLUE: [
        # Blue gradient (lighter)
        "#1E3A5C", "#234878", "#285594", "#2D63B0",
        "#3272C2", "#4385BE", "#66A0C8", "#92BFDB",
        "#ABCFE2", "#C6DDE8", "#E1ECEB", "#F0F5F4",
        # Red gradient (lighter)
        "#FFE1D5", "#FFCABB", "#FDB2A2", "#F89A8A",
        "#E8705F", "#D14D41", "#C03E35", "#AF3029",
        "#A02D27", "#913026", "#823424", "#733823",
    ],
    ColorScheme.CYAN: [
        "#101F1D", "#122F2C", "#143F3C", "#164F4A",
        "#1C6C66", "#24837B", "#2F968D", "#3AA99F",
        "#5ABDAC", "#87D3C3", "#A2DECE", "#BFE8D9",
        "#DDF1E4",
    ],
    ColorScheme.BLUE: [
        "#101A24", "#133051", "#163B66", "#1A4F8C",
        "#205EA6", "#3171B2", "#4385BE", "#66A0C8",
        "#92BFDB", "#ABCFE2", "#C6DDE8", "#E1ECEB",
    ],
    ColorScheme.RED: [
        "#261312", "#551B18", "#6C201C", "#942822",
        "#AF3029", "#C03E35", "#D14D41", "#E8705F",
        "#F89A8A", "#FDB2A2", "#FFCABB", "#FFE1D5",
    ],
    ColorScheme.BASE: [
        "#1C1B1A", "#282726", "#343331", "#403E3C",
        "#575653", "#6F6E69", "#878580", "#9F9D96",
        "#B7B5AC", "#CECDC3", "#DAD8CE", "#E6E4D9",
        "#F2F0E5",
    ],
    ColorScheme.YELLOW: [
        "#241E08", "#3A2D04", "#583D02", "#664D01",
        "#8E6801", "#AD8301", "#BE9207", "#D8A215",
        "#DFB431", "#ECCB60", "#F1D67E", "#F6E2A8",
        "#FAEEC6",
    ],
    ColorScheme.PURPLE: [
        "#1A1623", "#1A1623", "#261C39", "#31234E",
        "#3C2A62", "#5E409D", "#735EB5", "#8B7EC8",
        "#A699D0", "#C4B9E0", "#D3CAE6", "#E2D9E9",
        "#F0EAEC",
    ],
    ColorScheme.GREEN: [
        "#1A1E0C", "#252D09", "#313D07", "#3D4C07",
        "#536907", "#668008", "#768D21", "#879A39",
        "#A8AF54", "#BEC97E", "#CDD597", "#DDE2B2",
        "#EDEECF",
    ],
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def get_color_for_value(
    value: float, 
    min_value: float, 
    max_value: float, 
    scheme: ColorScheme
) -> Tuple[int, int, int]:
    """Get RGB color for a normalized value using the specified color scheme."""
    # Normalize the value to a 0-1 scale
    if max_value == min_value:
        normalized = 0.5
    else:
        normalized = (value - min_value) / (max_value - min_value)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
    
    # Get the appropriate color scheme
    colors = COLOR_SCHEMES[scheme]
    
    # Calculate the floating point index and interpolate between colors
    float_index = normalized * (len(colors) - 1)
    lower_index = int(float_index)
    upper_index = min(lower_index + 1, len(colors) - 1)
    
    # Get the two colors to interpolate between
    color1 = hex_to_rgb(colors[lower_index])
    color2 = hex_to_rgb(colors[upper_index])
    
    # Calculate the interpolation weight
    weight = float_index - lower_index
    
    # Interpolate between the two colors
    return tuple(
        int(c1 * (1 - weight) + c2 * weight)
        for c1, c2 in zip(color1, color2)
    ) 