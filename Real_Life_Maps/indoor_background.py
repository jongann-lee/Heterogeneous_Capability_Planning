"""Load the fine indoor floor plan used by evaluation rendering."""

from pathlib import Path

import numpy as np
from scipy import ndimage

FOUR_CONNECTED = ndimage.generate_binary_structure(2, 1)


def load_walkable(input_dir: Path) -> tuple[np.ndarray, dict]:
    """Keep the main interior plus explicitly labelled interior islands.

    The source uses 254 for free and 0 for wall. ``valid_space`` is a zone-ID
    raster, not a binary free-space mask: some passable corridors have ID 0.
    The largest four-connected free component is the primary interior. Positive
    zone IDs retain labelled rooms disconnected by closed/missing door pixels.
    Unlabelled exterior/background free components are discarded.
    """
    occupancy = np.load(input_dir / "occ_map.npy", allow_pickle=False)
    zones = np.load(input_dir / "valid_space.npy", allow_pickle=False)
    if occupancy.ndim != 2 or occupancy.shape != zones.shape:
        raise ValueError("occupancy and zone rasters must be aligned 2D arrays")
    values = set(map(int, np.unique(occupancy)))
    if values != {0, 254}:
        raise ValueError(f"expected occupancy values {{0, 254}}, got {values}")
    free = occupancy == 254
    labels, _ = ndimage.label(free, FOUR_CONNECTED)
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    main_component = int(component_sizes.argmax())
    walkable = (labels == main_component) | (free & (zones > 0))
    metadata = {
        "fine_shape": list(map(int, occupancy.shape)),
        "occupancy_free_value": 254,
        "occupancy_wall_value": 0,
        "main_free_component_pixels": int(component_sizes[main_component]),
        "walkable_pixels": int(walkable.sum()),
        "positive_zone_ids": [int(x) for x in np.unique(zones) if x > 0],
        "selection": "main 4-connected free component plus labelled free islands",
    }
    return np.ascontiguousarray(walkable, dtype=np.uint8), metadata
