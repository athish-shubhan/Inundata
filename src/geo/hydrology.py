"""DEM hydrology: depression filling, D8 flow routing, flow accumulation and HAND.

Height Above Nearest Drainage (Nobre et al. 2011) is the vertical distance from a
cell down to the stream cell its water actually flows into. That differs from
"elevation minus the lowest river point": a terrace 2 m above a side channel has
low HAND even if the main river is far below it, which is what controls whether
it floods.

Pure numpy plus one heap; the AOI (about 400k cells with the routing buffer)
runs in a few seconds, so a compiled dependency such as pysheds/numba is not
worth its install footprint here.
"""
import heapq

import numpy as np

OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_depressions(dem: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Priority-Flood with epsilon (Barnes, Lehman and Mulla 2014).

    Grows inward from the edges in order of elevation; any cell lower than the
    cell that reached it is raised just above it. The result drains everywhere
    to the border, with no pits and no perfectly flat areas, so D8 always has a
    strictly downhill neighbour.
    """
    h, w = dem.shape
    z = dem.astype("float64").ravel().copy()
    closed = np.isnan(z)
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    seeds = np.flatnonzero(edge.ravel() & ~closed)
    closed[seeds] = True
    heap = [(z[i], int(i)) for i in seeds]
    heapq.heapify(heap)
    while heap:
        zi, i = heapq.heappop(heap)
        r, c = divmod(i, w)
        for dr, dc in OFFSETS:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            j = rr * w + cc
            if closed[j]:
                continue
            closed[j] = True
            if z[j] <= zi:
                z[j] = zi + eps
            heapq.heappush(heap, (z[j], j))
    return z.reshape(h, w)


def d8_downstream(filled: np.ndarray, res: float) -> np.ndarray:
    """Flat index of each cell's steepest-descent neighbour, or -1 for outlets."""
    h, w = filled.shape
    pad = np.pad(filled, 1, constant_values=np.inf)
    rows, cols = np.mgrid[0:h, 0:w]
    best_slope = np.zeros((h, w))
    best = np.full((h, w), -1, dtype=np.int64)
    for dr, dc in OFFSETS:
        nb = pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        with np.errstate(invalid="ignore"):
            slope = (filled - nb) / (res * np.hypot(dr, dc))
            better = slope > best_slope
        best_slope = np.where(better, slope, best_slope)
        best = np.where(better, (rows + dr) * w + (cols + dc), best)
    return best.ravel()


def flow_accumulation(filled: np.ndarray, downstream: np.ndarray) -> np.ndarray:
    """Number of cells draining through each cell (itself included).

    Water only moves to strictly lower cells on the filled surface, so visiting
    cells from highest to lowest is a valid topological order.
    """
    acc = np.ones(filled.size)
    for i in np.argsort(filled.ravel(), kind="stable")[::-1]:
        j = downstream[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(filled.shape)


def height_above_drainage(dem: np.ndarray, filled: np.ndarray, downstream: np.ndarray,
                          drainage: np.ndarray) -> np.ndarray:
    """HAND: elevation minus the elevation of the drainage cell reached by following D8 flow."""
    z = dem.ravel()
    is_drain = drainage.ravel()
    base = np.full(z.size, np.nan)
    for i in np.argsort(filled.ravel(), kind="stable"):  # lowest first, so downstream is always resolved
        if is_drain[i]:
            base[i] = z[i]
        elif downstream[i] >= 0:
            base[i] = base[downstream[i]]
    return np.clip(z - base, 0, None).reshape(dem.shape)


def hand_from_dem(dem: np.ndarray, res: float, river: np.ndarray, channel_area_m2: float = 1e6,
                  burn_m: float = 5.0) -> dict[str, np.ndarray]:
    """Full chain. The mapped river is burned into the DEM so routing follows the
    real channel, and any cell draining more than ``channel_area_m2`` also counts
    as drainage so tributaries get their own baseline."""
    burned = np.where(river, dem - burn_m, dem)
    filled = fill_depressions(burned)
    down = d8_downstream(filled, res)
    acc = flow_accumulation(filled, down)
    drainage = river | (acc * res * res >= channel_area_m2)
    hand = height_above_drainage(dem, filled, down, drainage)
    return {"hand": hand.astype("float32"), "flow_acc": acc.astype("float32"), "drainage": drainage}
