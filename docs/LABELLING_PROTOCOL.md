# Tuesday reference sample: labelling protocol (draft v1)

Files: `reference_points.csv` (400 points, shuffled order, `label` column empty),
`gold_windows.csv` (5 windows of 50 x 50 cm, 1213 px), `repeat_subset_ids.txt` (40 ids),
`sampling_manifest.json` (seed 20261119).

QGIS: Layer > Add Delimited Text Layer, X = `x_m`, Y = `y_m`, CRS = the CRS of Tuesday.tif.
Work in id order P0001, P0002, ... Do not sort by position and do not look at any model output.

1. For each point, view a window of about 40 x 40 cm around it and label what is **under the
   point itself** (not the dominant class of the window).
2. Use only classes from the schema (`label_schema_v1_draft.csv`, after you approve it).
   If you cannot decide, write `UNSURE` and add a note; never guess silently.
3. If the point falls on a seam, hole, blur or stitching artefact, write `UNSCORABLE`.
   Do not skip it: these are counted and reported.
4. Mobile animals under the point (starfish, urchin, fish) are labelled as their own class;
   whether they count in cover is decided at analysis time, not while labelling.
5. After at least 7 days, relabel the 40 ids in `repeat_subset_ids.txt` blind (without
   looking at your first label). Agreement between the two passes is the upper bound on
   what any model can reach.
6. Gold windows are labelled last, as full polygons (no gaps), with the same schema.
