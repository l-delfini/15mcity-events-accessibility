
"""
CityScoreToolkit.py
--------------------------------------
Adds a scoring toggle so you can compute CityScore from:
  - "weighted" (default): original behavior using isochrone time_in_min weights (constant 15 in this setup)
  - "steps": new behavior using hex-hop distances per category (1 / (1 + steps)^alpha)

Also updates `get_pois` to accept either a file path or a preloaded GeoDataFrame.
Preserves earlier fixes: initialize `cs` early, keep *_steps, guard presence columns, align CRS.
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx
import osmnx as ox
import tobler

ScoreFrom = Literal["weighted", "steps"]

# -----------------------------
# Helpers
# -----------------------------

def set_osmnx_cache(folder: Optional[str] = None, use_cache: bool = True, log_console: bool = False):
    """
    Configure OSMnx cache to a writable folder (useful on read-only systems).
    """
    if folder is None:
        folder = str((Path.home() / ".osmnx_cache").resolve())
    os.makedirs(folder, exist_ok=True)
    ox.settings.cache_folder = folder
    ox.settings.use_cache = use_cache
    ox.settings.log_console = log_console
    return folder


# -----------------------------
# Core steps
# -----------------------------

def ISO(input_file_path: str,
        resolution: int = 9,
        travel_speed: float = 4.5,
        minutes: int = 15,
        mode: str = "walk"
        ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Build H3 grid over AOI, create network, and compute isochrones for each hex centroid.

    Returns:
        (AOI_wgs84, AOI_grid_3857, isochrones_3857)
    """
    # 1) read area of interest (AOI) and set the crs to 4326 if missing
    AOI1 = gpd.read_file(input_file_path)
    if AOI1.crs is None:
        AOI1 = AOI1.set_crs(4326)
    AOI1 = AOI1.to_crs(4326)

    # 2) Build H3 grid over AOI (returns hex polygons as GeoDataFrame)
    AOI = tobler.util.h3fy(AOI1, resolution=resolution, clip=False, buffer=False, return_geoms=True)
    AOI = gpd.GeoDataFrame(AOI, geometry="geometry", crs="EPSG:4326")
    AOI = AOI.reset_index(drop=True)
    # Create a stable hex id for joins
    if "hex_id" not in AOI.columns:
        AOI["hex_id"] = np.arange(len(AOI))

    # Metric version for measurements
    AOI_m = AOI.to_crs(3857)
    # Side length (approximate; perimeter/6 works well for regular hex)
    AOI_m["side_len_m"] = AOI_m.length / 6.0
    # Centroids in 3857
    AOI_m["centroid"] = AOI_m.geometry.centroid
    centroids_m = AOI_m[["hex_id", "centroid"]].copy()
    centroids_m = gpd.GeoDataFrame(centroids_m, geometry="centroid", crs=3857)

    # 3) Build network from AOI bounds (WGS84 bbox)
    west, south, east, north = AOI1.total_bounds  # (minx, miny, maxx, maxy)
    G = ox.graph_from_bbox(north, south, east, west, network_type=mode, simplify=False)

    # 4) Project graph to metric CRS
    Gp = ox.project_graph(G, to_crs=3857, to_latlong=False)

    # 5) Add travel time (minutes) to edges
    meters_per_min = (travel_speed * 1000.0) / 60.0
    for _, _, _, data in Gp.edges(data=True, keys=True):
        length = float(data.get("length", 0.0))
        data["time"] = (length / meters_per_min) if meters_per_min > 0 else float("inf")

    # 6) Snap centroids to nearest nodes (allow distance up to 1.5 * side length)
    X = centroids_m["centroid"].x.values
    Y = centroids_m["centroid"].y.values
    nearest_nodes, distances = ox.nearest_nodes(Gp, X, Y, return_dist=True)

    snap_df = pd.DataFrame({
        "hex_id": centroids_m["hex_id"].values,
        "node": nearest_nodes,
        "dist_m": distances
    })
    snap_df = snap_df.merge(AOI_m[["hex_id", "side_len_m"]], on="hex_id", how="left")
    snap_df = snap_df[snap_df["dist_m"] <= (1.5 * snap_df["side_len_m"])]
    if snap_df.empty:
        # No snaps → return empty isochrones; grid is still valid
        return AOI1, AOI_m.drop(columns=["centroid"]), gpd.GeoDataFrame(geometry=[], crs=3857)

    # 7) Build isochrones per snapped centroid
    rows = []
    for _, r in snap_df.iterrows():
        n = r["node"]
        hid = r["hex_id"]
        # ego-graph within time radius
        sub = nx.ego_graph(Gp, n, radius=float(minutes), distance="time")
        if sub.number_of_nodes() == 0:
            continue
        # convex hull of nodes
        pts = [Point(d["x"], d["y"]) for _, d in sub.nodes(data=True)]
        hull = gpd.GeoSeries(pts, crs=3857).unary_union.convex_hull
        rows.append({"grid_id": hid, "time_in_min": minutes, "geometry": hull})

    if not rows:
        return AOI1, AOI_m.drop(columns=["centroid"]), gpd.GeoDataFrame(geometry=[], crs=3857)

    isochrones = gpd.GeoDataFrame(rows, geometry="geometry", crs=3857)

    # Return AOI (WGS84), grid (3857), isochrones (3857)
    return AOI1, AOI_m.drop(columns=["centroid"]), isochrones


def get_pois(AOI: gpd.GeoDataFrame, official_pois: Optional[str] = None,
             official_gdf: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
    """
    Return a POI GeoDataFrame in EPSG:3857 matching the AOI grid CRS.
    Accepts either a file path or a preloaded GeoDataFrame.
    """
    if official_gdf is not None:
        off = official_gdf
    elif official_pois is not None:
        off = gpd.read_file(official_pois)
    else:
        raise ValueError("Provide either official_pois path or official_gdf GeoDataFrame.")

    if off.crs is None:
        off = off.set_crs(4326)
    off = off.to_crs(3857)
    return off


# -----------------------------
# NEW: Hex-steps helpers
# -----------------------------

def _hex_adjacency_graph(AOI_hex_3857: gpd.GeoDataFrame) -> nx.Graph:
    """
    Build Queen-adjacency graph over hex grid (EPSG:3857).
    Requires a 'hex_id' column and polygon geometries.
    """
    if "hex_id" not in AOI_hex_3857.columns:
        AOI_hex_3857 = AOI_hex_3857.copy()
        AOI_hex_3857["hex_id"] = np.arange(len(AOI_hex_3857))

    sindex = AOI_hex_3857.sindex
    G = nx.Graph()
    hex_ids = AOI_hex_3857["hex_id"].to_numpy()
    geoms = AOI_hex_3857.geometry.to_numpy()

    for hid, geom in zip(hex_ids, geoms):
        G.add_node(hid)
        # candidate neighbors via bbox hits
        for j in sindex.intersection(geom.bounds):
            hid2 = AOI_hex_3857["hex_id"].iloc[j]
            if hid2 == hid:
                continue
            geom2 = AOI_hex_3857.geometry.iloc[j]
            # Queen adjacency (touches), robust with small buffer(0)
            if geom.touches(geom2) or geom.buffer(0).touches(geom2.buffer(0)):
                G.add_edge(hid, hid2)
    return G


def _category_steps_from_pois(AOI_hex_3857: gpd.GeoDataFrame,
                              POIS_3857: gpd.GeoDataFrame,
                              *,
                              minutes: int = 15,
                              speed_kmh: float = 4.5,
                              steps_cap_from_minutes: bool = True) -> pd.DataFrame:
    """
    Returns a wide DataFrame with columns: ['hex_id', '<cat>_steps'].
    Steps = minimum hex hops from each hex to nearest POI of that category.
    Optionally computes a hop cap equivalent to a 15-min radius (but does not drop values).
    """
    # 1) Build adjacency graph once
    G = _hex_adjacency_graph(AOI_hex_3857)

    # 2) Assign POIs to hexes (same CRS)
    if POIS_3857.crs != AOI_hex_3857.crs:
        POIS_3857 = POIS_3857.to_crs(AOI_hex_3857.crs)

    poi_hex = gpd.sjoin(
        POIS_3857[["category", "geometry"]],
        AOI_hex_3857[["hex_id", "geometry"]],
        how="inner",
        predicate="within"
    )[["category", "hex_id"]].drop_duplicates()

    # If no POIs, return only hex_id column
    if poi_hex.empty:
        return pd.DataFrame({"hex_id": AOI_hex_3857["hex_id"].values})

    # 3) Optional hop budget to mirror ~15 min
    meters_per_min = (speed_kmh * 1000.0) / 60.0
    if "side_len_m" in AOI_hex_3857.columns:
        step_len = 1.5 * float(AOI_hex_3857["side_len_m"].median())
    else:
        # crude fallback
        step_len = float(np.sqrt(AOI_hex_3857.geometry.area.median())) * 1.2
    steps_max = int(np.floor((minutes * meters_per_min) / step_len)) if steps_cap_from_minutes else None

    # 4) Multi-source BFS per category
    all_hex_ids = AOI_hex_3857["hex_id"].values
    out_cols = {"hex_id": all_hex_ids.copy()}
    from collections import deque

    for cat, grp in poi_hex.groupby("category"):
        sources = list(grp["hex_id"].unique())
        if not sources:
            continue
        dist = {s: 0 for s in sources}
        dq = deque(sources)
        visited = set(sources)

        while dq:
            u = dq.popleft()
            for v in G.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    dist[v] = dist[u] + 1
                    dq.append(v)

        steps = [dist.get(h, np.nan) for h in all_hex_ids]
        out_cols[f"{cat}_steps"] = steps

    df = pd.DataFrame(out_cols)
    if steps_max is not None:
        df.attrs["steps_max"] = steps_max
    return df


# -----------------------------
# Scoring / intersections
# -----------------------------

def make_intersection(
    AOI_hex: gpd.GeoDataFrame,
    POIS: gpd.GeoDataFrame,
    gdf: Optional[gpd.GeoDataFrame] = None,
    *,
    alpha: float = 0.08,
    max_categories: int = 5,
    # toggle: compute hex-hop distances per category
    add_hex_steps: bool = True,
    minutes_for_steps: int = 15,
    speed_kmh_for_steps: float = 4.5,
    # NEW: choose how to compute CityScore ("weighted" or "steps")
    score_from: ScoreFrom = "weighted"
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Intersect isochrones with POIs and aggregate to hex grid.
    Outputs:
      reachable_pois, <cat>_raw, <cat>_weighted, <cat>_presence,
      initial_cityscore, coexistence, cityscore,
      and (NEW) <cat>_steps if add_hex_steps=True.

    Notes:
      - 'score_from="weighted"' keeps original behavior (weights from isochrone time_in_min).
      - 'score_from="steps"' sets weights as 1 / (1 + steps)^alpha and recomputes CityScore.
      - We initialize 'cs' early as a copy of AOI_hex and always merge into it,
        so earlier diagnostic columns (e.g., *_steps) are preserved.
      - Presence columns are set from weighted only if not already present,
        avoiding clobbering a steps-based presence.
    """
    # Ensure hex_id exists
    if "hex_id" not in AOI_hex.columns:
        AOI_hex = AOI_hex.copy()
        AOI_hex["hex_id"] = np.arange(len(AOI_hex))

    # Initialize output frame early and preserve columns via merges
    cs = AOI_hex.copy()

    # If no isochrones, we can still compute steps; other metrics go to zero
    if gdf is None or gdf.empty:
        # Add steps (optional)
        if add_hex_steps:
            AOI_use = AOI_hex if AOI_hex.crs == 3857 else AOI_hex.to_crs(3857)
            POIS_use = POIS if POIS.crs == 3857 else POIS.to_crs(3857)
            steps_df = _category_steps_from_pois(
                AOI_use, POIS_use,
                minutes=minutes_for_steps,
                speed_kmh=speed_kmh_for_steps,
                steps_cap_from_minutes=True
            )
            cs = cs.merge(steps_df, on="hex_id", how="left")

            # If scoring from steps, compute weights & final score even without isochrones
            if score_from == "steps":
                step_cols = [c for c in cs.columns if c.endswith("_steps")]
                weighted_cols = []
                for col in step_cols:
                    base = col[:-6]
                    wcol = f"{base}_weighted"
                    reach_col = f"{base}_reachable"
                    cs[wcol] = np.where(cs[col].notna(), 1.0 / np.power(1.0 + cs[col], alpha), 0.0)
                    if reach_col in cs.columns:
                        cs[wcol] = np.where(cs[reach_col] == 1, cs[wcol], 0.0)
                    weighted_cols.append(wcol)
                cs["initial_cityscore"] = cs[weighted_cols].sum(axis=1) if weighted_cols else 0.0
                cs["cityscore"] = ((cs["initial_cityscore"] / float(max_categories)) * 100.0).clip(0.0, 100.0)
                # steps-based presence (≤ hop budget) and coexistence (fallback to steps_max if present)
                steps_max = steps_df.attrs.get("steps_max", None)
                if steps_max is not None:
                    for col in step_cols:
                        base = col[:-6]
                        pres_col = f"{base}_presence"
                        if pres_col not in cs.columns:
                            cs[pres_col] = ((cs[col].notna()) & (cs[col] <= steps_max)).astype(int)
                    presence_cols = [c for c in cs.columns if c.endswith("_presence")]
                    cs["coexistence"] = cs[presence_cols].sum(axis=1) if presence_cols else 0
                else:
                    cs["coexistence"] = 0
            else:
                cs["initial_cityscore"] = 0.0
                cs["cityscore"] = 0.0
                cs["coexistence"] = 0

        else:
            cs["reachable_pois"] = 0
            cs["initial_cityscore"] = 0.0
            cs["coexistence"] = 0
            cs["cityscore"] = 0.0

        diagnostic = gpd.GeoDataFrame(geometry=[], crs=AOI_hex.crs)
        # Ensure *_raw columns exist (zeros) even on empty/no-reach days
        if "category" in POIS.columns:
            cats = sorted(pd.Series(POIS["category"]).dropna().unique())
            for cat in cats:
                rc = f"{cat}_raw"
                if rc not in cs.columns:
                    cs[rc] = 0
        for name in cs.columns:
            if name.endswith("_raw"):
                cs[name] = cs[name].fillna(0).astype(int)
        return cs, diagnostic

    # Align CRS with isochrones
    if POIS.crs != gdf.crs:
        POIS = POIS.to_crs(gdf.crs)
    if AOI_hex.crs != gdf.crs:
        AOI_hex = AOI_hex.to_crs(gdf.crs)
        cs = cs.to_crs(gdf.crs)  # keep cs aligned too

    # --- (A) Optional: discrete hex-hop distances (per category) ---
    if add_hex_steps:
        AOI_use = AOI_hex if AOI_hex.crs == 3857 else AOI_hex.to_crs(3857)
        POIS_use = POIS if POIS.crs == 3857 else POIS.to_crs(3857)
        steps_df = _category_steps_from_pois(
            AOI_use, POIS_use,
            minutes=minutes_for_steps,
            speed_kmh=speed_kmh_for_steps,
            steps_cap_from_minutes=True
        )
        cs = cs.merge(steps_df, on="hex_id", how="left")

        # Optional: derive presence from steps (≤ hop budget) without clobbering later presence
        steps_max = steps_df.attrs.get("steps_max", None)
        if steps_max is not None:
            for col in [c for c in cs.columns if c.endswith("_steps")]:
                base = col[:-6]
                pres_col = f"{base}_presence"
                if pres_col not in cs.columns:
                    cs[pres_col] = ((cs[col].notna()) & (cs[col] <= steps_max)).astype(int)

    # --- (B) reachable_pois (raw) ---
    iso_diss = gdf[["grid_id", "geometry"]].dissolve(by="grid_id", as_index=False)
    join_within = gpd.sjoin(POIS, iso_diss[["grid_id", "geometry"]], how="inner", predicate="within")
    counts = join_within.groupby("grid_id").size().rename("reachable_pois").reset_index()

    cs = cs.merge(counts, left_on="hex_id", right_on="grid_id", how="left")
    cs["reachable_pois"] = cs["reachable_pois"].fillna(0).astype(int)
    cs = cs.drop(columns=["grid_id"], errors="ignore")


    # --- Build per-category reachability inside isochrone ---
    reach_cat = gpd.sjoin(
        POIS[["category", "geometry"]],
        iso_diss[["grid_id", "geometry"]],
        how="inner",
        predicate="within"
    )
    if not reach_cat.empty:
        reach_pivot = (reach_cat
                       .groupby(["grid_id", "category"])
                       .size()
                       .unstack(fill_value=0)
                       .reset_index())
        # counts -> binary reachability
        for c in list(reach_pivot.columns):
            if c != "grid_id":
                reach_pivot[c] = (reach_pivot[c] > 0).astype(int)
        # rename to *_reachable and merge onto cs
        reach_pivot = reach_pivot.rename(columns={c: (f"{c}_reachable" if c != "grid_id" else c)
                                                  for c in reach_pivot.columns})
        cs = cs.merge(reach_pivot, left_on="hex_id", right_on="grid_id", how="left")
        cs = cs.drop(columns=["grid_id"], errors="ignore")
        for c in list(cs.columns):
            if c.endswith("_reachable"):
                cs[c] = cs[c].fillna(0).astype(int)
        # --- (C) weighted (closest time), presence, scoring ---
    touch = gpd.sjoin(
        gdf[["grid_id", "time_in_min", "geometry"]],
        POIS[["category", "geometry"]],
        how="inner"
    )
    if touch.empty:
        # If scoring from steps, finalize here from steps; else keep zeros.
        if score_from == "steps":
            step_cols = [c for c in cs.columns if c.endswith("_steps")]
            weighted_cols = []
            for col in step_cols:
                base = col[:-6]
                wcol = f"{base}_weighted"
                cs[wcol] = np.where(cs[col].notna(), 1.0 / np.power(1.0 + cs[col], alpha), 0.0)
                weighted_cols.append(wcol)
            cs["initial_cityscore"] = cs[weighted_cols].sum(axis=1) if weighted_cols else 0.0
            cs["cityscore"] = ((cs["initial_cityscore"] / float(max_categories)) * 100.0).clip(0.0, 100.0)
            presence_cols = [c for c in cs.columns if c.endswith("_presence")]
            cs["coexistence"] = cs[presence_cols].sum(axis=1) if presence_cols else 0
        else:
            cs["initial_cityscore"] = cs.get("initial_cityscore", 0.0)
            cs["coexistence"] = cs.get("coexistence", 0)
            cs["cityscore"] = cs.get("cityscore", 0.0)
    else:
        touch = touch.sort_values("time_in_min")
        nearest = touch.groupby(["grid_id", "category"], as_index=False).first()

        # --- Compute "weighted" weights from time_in_min (always available) ---
        w = nearest["time_in_min"].to_numpy(dtype=float)
        weights = np.where(w <= 0.0, 1.0, 1.0 / np.power(w, alpha))
        nearest = nearest.assign(weight=weights)

        weighted = (nearest
                    .pivot(index="grid_id", columns="category", values="weight")
                    .fillna(0.0)
                    .reset_index())
        weighted.columns.name = None
        # rename to *_weighted
        new_cols = {}
        for name in weighted.columns:
            if name != "grid_id":
                new_cols[name] = f"{name}_weighted"
        weighted = weighted.rename(columns=new_cols)

        cs = cs.merge(weighted, left_on="hex_id", right_on="grid_id", how="left")
        cs = cs.drop(columns=["grid_id"], errors="ignore")
        for name in cs.columns:
            if name.endswith("_weighted"):
                cs[name] = cs[name].fillna(0.0)

        # raw per-category counts
        within_cat = gpd.sjoin(
            POIS[["category", "geometry"]],
            iso_diss[["grid_id", "geometry"]],
            how="inner",
            predicate="within"
        )
        if not within_cat.empty:
            raw = (within_cat
                   .groupby(["grid_id", "category"])
                   .size()
                   .unstack(fill_value=0)
                   .reset_index())
            # rename to *_raw
            new_cols = {}
            for name in raw.columns:
                if name != "grid_id":
                    new_cols[name] = f"{name}_raw"
            raw = raw.rename(columns=new_cols)

            cs = cs.merge(raw, left_on="hex_id", right_on="grid_id", how="left")
            cs = cs.drop(columns=["grid_id"], errors="ignore")

        # fill *_raw ints
        for name in cs.columns:
            if name.endswith("_raw"):
                cs[name] = cs[name].fillna(0).astype(int)

        # presence from weighted (only if not already set, preserves steps-based presence)
        weighted_cols = [name for name in cs.columns if name.endswith("_weighted")]
        for name in weighted_cols:
            base = name[:-9]  # strip "_weighted"
            pres_col = f"{base}_presence"
            if pres_col not in cs.columns:  # guard to not clobber steps-based presence
                cs[pres_col] = (cs[name] > 0).astype(int)

        # --- Final scoring depending on score_from ---
        if score_from == "weighted":
            cs["initial_cityscore"] = cs[weighted_cols].sum(axis=1) if weighted_cols else 0.0
        elif score_from == "steps":
            # compute weights from steps: 1 / (1 + steps)^alpha, but zero if not reachable in isochrone
            step_cols = [c for c in cs.columns if c.endswith("_steps")]
            step_weighted_cols = []
            for col in step_cols:
                base = col[:-6]
                wcol = f"{base}_weighted"
                reach_col = f"{base}_reachable"
                cs[wcol] = np.where(cs[col].notna(), 1.0 / np.power(1.0 + cs[col], alpha), 0.0)
                if reach_col in cs.columns:
                    cs[wcol] = np.where(cs[reach_col] == 1, cs[wcol], 0.0)
                step_weighted_cols.append(wcol)
            # Use ONLY steps-based weights to avoid double counting
            cs["initial_cityscore"] = cs[step_weighted_cols].sum(axis=1) if step_weighted_cols else 0.0
        else:
            raise ValueError(f"Unknown score_from: {score_from}")

        presence_cols = [name for name in cs.columns if name.endswith("_presence")]
        cs["coexistence"] = cs[presence_cols].sum(axis=1) if presence_cols else 0
        cs["cityscore"] = (
            ((cs["initial_cityscore"] / float(max_categories)) * 100.0)
            .clip(lower=0.0, upper=100.0)
            if max_categories > 0 else 0.0
        )

    # --- (D) diagnostic (counts and % by category inside hex) ---
    diag_join = gpd.sjoin(
        AOI_hex[["hex_id", "geometry"]],
        POIS[["category", "geometry"]],
        how="left", predicate="intersects"
    )
    if diag_join.empty:
        diagnostic = gpd.GeoDataFrame(geometry=[], crs=AOI_hex.crs)
    else:
        grp = diag_join.groupby(["hex_id", "category"]).size().rename("count").reset_index()
        totals = grp.groupby("hex_id")["count"].sum().rename("total_count").reset_index()
        diag = grp.merge(totals, on="hex_id", how="left")
        diag["percentage"] = np.where(diag["total_count"] > 0,
                                      100.0 * diag["count"] / diag["total_count"], 0.0)
        wide_cnt = diag.pivot(index="hex_id", columns="category", values="count").fillna(0).add_suffix("_count")
        wide_pct = diag.pivot(index="hex_id", columns="category", values="percentage").fillna(0.0).add_suffix("_pct")

        diagnostic = AOI_hex.merge(wide_cnt.reset_index(), on="hex_id", how="left")
        diagnostic = diagnostic.merge(wide_pct.reset_index(), on="hex_id", how="left")
        for name in diagnostic.columns:
            if name.endswith("_count"):
                diagnostic[name] = diagnostic[name].fillna(0).astype(int)
            elif name.endswith("_pct"):
                diagnostic[name] = diagnostic[name].fillna(0.0)

    # Ensure *_raw columns exist (zeros) even on empty/no-reach days
    if "category" in POIS.columns:
        cats = sorted(pd.Series(POIS["category"]).dropna().unique())
        for cat in cats:
            rc = f"{cat}_raw"
            if rc not in cs.columns:
                cs[rc] = 0
    for name in cs.columns:
        if name.endswith("_raw"):
            cs[name] = cs[name].fillna(0).astype(int)
    
    # === IN-HEX PRESENCE / COEXISTENCE (authoritative) ===
    # Presence is defined strictly by *containment* of POIs within the AOI hex polygon.
    #   X_presence = 1 iff there is at least one event of category X inside that hex; 0 otherwise.
    #   coexistence = number of distinct categories present in the hex (sum of *_presence).
    try:
        POIS_on_AOI = POIS if (POIS.crs == AOI_hex.crs) else POIS.to_crs(AOI_hex.crs)

        join_inhex = gpd.sjoin(
            POIS_on_AOI[["category", "geometry"]],
            AOI_hex[["hex_id", "geometry"]],
            how="inner",
            predicate="within",
        )

        # Build a wide presence table (hex_id + one *_presence column per category)
        if not join_inhex.empty:
            cnt_inhex = (
                join_inhex.groupby(["hex_id", "category"])
                .size().rename("count").reset_index()
            )
            wide_inhex = (
                cnt_inhex.pivot(index="hex_id", columns="category", values="count")
                .fillna(0).astype(int).reset_index()
            )
            # counts -> binary presence
            for c in list(wide_inhex.columns):
                if c != "hex_id":
                    wide_inhex[c] = (wide_inhex[c] > 0).astype(int)
            # rename to *_presence
            wide_inhex = wide_inhex.rename(
                columns={c: (f"{c}_presence" if c != "hex_id" else c) for c in wide_inhex.columns}
            )

            # Ensure every hex_id is present (fill missing with zeros)
            all_hex = pd.DataFrame({"hex_id": cs["hex_id"].values})
            wide_inhex_full = all_hex.merge(wide_inhex, on="hex_id", how="left").fillna(0)
            for c in list(wide_inhex_full.columns):
                if isinstance(c, str) and c.endswith("_presence"):
                    wide_inhex_full[c] = wide_inhex_full[c].astype(int)

            # Drop any existing *_presence columns to avoid _x/_y duplication, then merge cleanly
            pres_cols_existing = [c for c in cs.columns if isinstance(c, str) and c.endswith("_presence")]
            if pres_cols_existing:
                cs = cs.drop(columns=pres_cols_existing, errors="ignore")

            cs = cs.merge(wide_inhex_full, on="hex_id", how="left")

        else:
            # No POIs inside any hex: drop previous presence cols (if any)
            pres_cols_existing = [c for c in cs.columns if isinstance(c, str) and c.endswith("_presence")]
            if pres_cols_existing:
                cs = cs.drop(columns=pres_cols_existing, errors="ignore")

        # Recompute coexistence from the (new) *_presence columns
        _pres_cols = [c for c in cs.columns if isinstance(c, str) and c.endswith("_presence")]
        cs["coexistence"] = cs[_pres_cols].sum(axis=1) if _pres_cols else 0
    except Exception as _e:
        # If anything goes wrong, leave previous values; better to be permissive than to crash.
        pass

    # Drop any '*_raw' columns per your spec
    raw_cols = [c for c in cs.columns if isinstance(c, str) and c.endswith("_raw")]
    if raw_cols:
        cs = cs.drop(columns=raw_cols, errors="ignore")

    return cs, diagnostic


# -----------------------------
# CLI runner (optional)
# -----------------------------

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Plain-Python City Score runner (official-only by default).")
    parser.add_argument("--boundary", required=True, help="Path to AOI polygon file (shp/gpkg/geojson).")
    parser.add_argument("--pois", required=False, help="Path to POIs file (points). Required if --mode official or osm+official.")
    parser.add_argument("--mode", choices=["official", "osm", "osm+official"], default="official",
                        help="Which POI source to use.")
    parser.add_argument("--h3", type=int, default=9, help="H3 resolution (default 9).")
    parser.add_argument("--speed", type=float, default=4.5, help="Travel speed km/h (default 4.5).")
    parser.add_argument("--minutes", type=int, default=15, help="Isochrone cutoff minutes (default 15).")
    parser.add_argument("--network", choices=["walk", "bike", "drive", "all"], default="walk",
                        help="OSMnx network_type (default walk).")
    parser.add_argument("--score-from", choices=["weighted", "steps"], default="weighted",
                        help="How to compute CityScore: 'weighted' (original) or 'steps' (new).")
    parser.add_argument("--out", default="cityscore_outputs.gpkg", help="Output GeoPackage path.")

    args = parser.parse_args()

    # Cache folder (writable)
    set_osmnx_cache(use_cache=True, log_console=True)

    # ISO
    t0 = time.perf_counter()
    AOI_bounds, AOI_hex, isochrones = ISO(
        args.boundary,
        resolution=args.h3,
        travel_speed=args.speed,
        minutes=args.minutes,
        mode=args.network
    )

    # POIs
    if not args.pois:
        raise SystemExit("--pois is required (official POIs). OSM mode is not implemented in get_pois().")
    pois = get_pois(AOI_bounds, official_pois=args.pois)

    # Intersections (scored) with steps and scoring toggle
    CityScore, Diagnostic = make_intersection(
        AOI_hex, pois, gdf=isochrones,
        add_hex_steps=True,
        minutes_for_steps=args.minutes,
        speed_kmh_for_steps=args.speed,
        score_from=args.score_from
    )

    # Save
    out = Path(args.out)
    CityScore.to_file(out, layer="cityscore", driver="GPKG")
    Diagnostic.to_file(out, layer="diagnostic", driver="GPKG")
    dt = time.perf_counter() - t0
    print(f"✅ Done in {dt:.1f}s → {out}")


if __name__ == "__main__":
    _main()
