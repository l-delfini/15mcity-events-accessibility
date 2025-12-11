# CityScoreToolkit.py

This is a plain-Python port of the 15-minute City Score toolkit originally released by Fondazione Transform Transport ETS (QGIS script + notebook on Zenodo). It computes a **CityScore** for each hexagon in an H3 grid by measuring access to different categories of POIs within a 15-minute walk.

## What does it do?

- Builds an H3 grid over your area of interest.
- Uses OSMnx to create a walking network and compute isochrones (e.g. 15 minutes).
- Intersects isochrones with POIs (e.g. services, transport, green areas).
- Aggregates everything to hexes and outputs:
  - a CityScore per hex,
  - reachability and coexistence of categories,
  - optional diagnostic counts per category.

## What’s new compared to the original script?

This version adds a few things on top of the QGIS/Zenodo code:

- **Scoring toggle** (`score_from`):
  - `"weighted"` (default): original behavior using isochrone travel time (`time_in_min`) and a decay exponent `alpha`.
  - `"steps"`: new behavior using hex-hop distances per category  
    (`weight = 1 / (1 + steps)^alpha`), based on a hex adjacency graph.


You can run it as a script (CLI) or import the functions (`ISO`, `get_pois`, `make_intersection`) in your own analysis.
