# PCA Explorer

This report contract is adapted from Silico's PCA and embedding explorer for the
CausaLab exploratory phase. The PCA experiment is the only source of coordinates
for this report.

Build one self-contained HTML page where the researcher can navigate the PCA
projection and click any point to inspect the model input and selected token that
produced it.

This report displays residual stream vectors from language-model inputs. Keep the
general interaction rules below, but use PCA coordinates only.

## Generate concepts for coloring

For the given dataset, identify plausible concepts that might be reflected in the components. Use the researcher's description, available metadata, representative examples, and domain knowledge. Generate at least 5–6 concepts when the dataset supports them; do not stop after the first one or two obvious labels. Include a useful mix of categorical, continuous, count-based, and structural or temporal concepts.

Use these concepts as color-by options in the explorer so the researcher can recolor the same coordinates and inspect which candidate properties align with each component or visible cluster.

Examples include the changed input value, expected answer, model answer,
correctness, prompt region, token position, operand identity, entity identity, and
other task variables available in the behavioral dataset.

The fit does not become more expensive as more candidate concepts are proposed: fit it once, then color the same coordinates by each concept. Treat these concepts as hypotheses for visual inspection. A principal component maximizes variance; an apparent association does not by itself establish that the component represents or causally controls the concept.

## Compute once, ship the coordinates

Fit PCA once for each declared layer and token-position population. Save 2D and
3D coordinates for every point alongside metadata. Keep the top 10 components
and record variance explained per component.

The default axes should be PC1, PC2 and PC3 when we have no additional labels.
When labels are available, score each component against them. Use eta-squared for
a categorical label and Spearman correlation for an ordinal label. Record those
scores beside the variance and show the components that carry the signal.
A page that opens on a plane where the effect is invisible gives a false initial
impression when the structure is one axis away.
Say in the caption which components carry the label and which do not, with the scores.

**This page is cheap.** It is a browser build over already-computed coordinates, with no model, no GPU, and no harvest. When someone asks to see it, build it and show it in the same turn. Never queue a run for it, and never make a researcher ask twice.

Cap at a few thousand points before reducing. More than that makes the page slow and the plot unreadable. If subsampling, use a stratified sample so rare classes are not dropped.

Pack everything into one JSON blob embedded directly in the HTML: coordinates,
labels, colors, and the full input text and selected token for every point. The
page must make no network requests.

## Build the page

**Scatter plot.** Plotly.js `scatter` (2D) or `scatter3d` (3D). Embed Plotly inline in the HTML (`plotly.min.js` bundled into the file) so the page works with no network access — do not load from CDN. One trace per class so the legend is automatic. Points at 8px, opacity 0.75 in 2D so dense clusters don't become a solid blob. In 3D, points carry no marker alpha at all: depth comes from the halo and shadow layers below, and per-marker transparency muddies both.

**Axis dropdowns.** X and Y (and Z in 3D mode) pick any of the 10 saved components. Switching rerenders instantly from already-computed coordinates.

**Variance bar.** Add a small bar chart below the axis selectors, with one bar per
component and height equal to the percentage of variance explained. Without this,
the reader cannot tell whether PC1 captures 80% of the structure or 8%.

**Color.** If labels are available, color by label using a qualitative colormap. If there are no labels, default to RGB from the first three components: rescale PC1, PC2, PC3 each to [0, 1] across all points and use them as R, G, B. This makes the color reflect the embedding structure itself with no arbitrary choices. Add a toggle to switch between label color, PC-RGB color, and flat grey.

**Click to inspect.** Clicking a point opens a side panel on the right:
- Text source: show the full text in a monospace box, key token bold.
- List the 5 nearest neighbors by distance in the **original high-dimensional space** (not the projected space), with their labels. Clicking a neighbor moves the selection.

Panel slides in without pushing the plot. `Escape` closes it. URL updates with the point index so the view is shareable.

**Hover tooltip.** Label + one-line preview only. Keep it small — full detail is for clicking.

## Style

Same Goodfire palette as the other visualizers. Selected point gets an accent-colored ring. Nearest neighbors get a thinner muted ring.

**Class colors are fixed for the life of the thread.** Take them in order from the palette's categorical colorway and record the mapping in the page's JSON. Every later view of the same data, another site, another projection, a figure in the report, a diagram in chat, reuses that mapping. Two views of one dataset in two color schemes forces the researcher to re-learn the legend and reads as two different results. Do not reach for the palette's semantic hues (success, failure) for neutral peer categories.

## 3D mode

3D is the default view whenever three or more components are needed to show the
structure; 2D stays available behind a toggle. In 3D, use Plotly `scatter3d`
with the recipe below. The recipe has **four required trace types**, in this
order, plus an optional label trace. A page missing a required trace has not
implemented it:

1. **Floor**, a `surface` trace. A flat square (important that it should be a square) below the cloud at `z_floor`, slightly under the lowest point. Light grey, `showscale: false`, `hoverinfo: "skip"`, flat lighting. Side length equals the **largest** of the three axis ranges, centred on the cloud, so it reads square rather than stretched. The floor is its own geometry: the shadow points are not the floor, and a page with shadow points floating over nothing is the most common way this recipe is half-built.
2. **Shadow**, a `scatter3d` trace. Every point re-plotted at `z_floor`, flat mid-grey, trace-level `opacity 0.15`.
3. **Halo**, one `scatter3d` trace per point trace. Same colors, 3× the point size, trace-level `opacity 0.1`, `showlegend: false`, `hoverinfo: "skip"`. Build it and **push it**: a halo that is constructed and then dropped is the failure this line exists to prevent.
4. **Points**, opaque `rgb()` strings, 6px, borderless.
5. **Labels** (conditional): if every label string is ≤ 20 characters, add one `scatter3d` trace with `mode: "text"`, `text` set to the label strings, `textposition: "top center"`, small font (10px), same x/y/z as the points. Hidden by default (`visible: false`); shown via the Labels toggle.

Then the scene: axes hidden, and a camera placed from one radius and the two stated angles so the rendered elevation is what the spec says. Do not hand-tune a component:

```
eye = (r·cos(el)·cos(az), r·cos(el)·sin(az), r·sin(el))   el = 16°, az = −60°
```

Give all three axes an equal data span centred on the cloud and set `aspectmode` so the display box is a cube. Equal spans keep the data proportions honest; the cube makes the floor render square. Letting the raw axis ranges drive the aspect stretches the floor into a rectangle and is the other half of the "floor looks wrong" report.

- **Point-size slider** (0.5×–3×) via `Plotly.restyle` on point + halo traces (halo tracks at 3×). Add an **opacity slider** for the 2D case (drives CSS `opacity` of the trace layer — no re-render). Both sliders update instantly without touching the data.
- **Layer toggles** for halo, shadow and floor. They double as diagnostics: when the 3D view looks wrong, switching layers off localizes which one is at fault in seconds.
- **Labels toggle**: a button that calls `Plotly.restyle` to flip `visible` on the labels trace (only present when labels were added). Default: hidden.

### Verify by looking at the rendered page

A comment claiming the recipe was followed is not evidence, and neither is a diff. Screenshot the page as it lands and confirm, before handing it over:

- it opens in 3D on the components that carry the labeled structure;
- the floor is a `surface` trace, flat, below the cloud, with equal x and y sides matching the largest axis range;
- the shadow's z equals the floor's z exactly, at opacity 0.15;
- one halo per point trace exists in the figure, at 3× size and opacity 0.1, and still tracks at 3× after moving the size slider;
- recovered camera elevation is 16° and azimuth is −60°;
- the cloud is centred in the frame at a size that fills it, with the whole shadow landing on the floor;
- the layer toggles hide and restore all three layers;
- the labels toggle hides and restores the label trace (when present);
- the caption names which components carry the label.

## Pitfalls

**Neighbors in projected space are wrong.** The 2D picture distorts distances. Always compute neighbors in the original space and say so in the panel.

**A low-variance component can be the whole point.** Variance rank and label relevance are unrelated. The component that separates your classes may be PC3 or PC7 while PC1 carries norm or token count, so the leading-variance view can look like one undifferentiated cloud while the structure sits one axis over. Report the per-component label scores next to the variance so this is visible rather than inferred.

**Three components are a thin slice.** Print the participation ratio beside the axes. When it is in the dozens, say plainly that the view is a small projection of the geometry and that overlapping clouds do not imply the classes are inseparable in the full space.

**Too many points.** Above ~5000, Plotly's SVG backend lags. Switch to `scattergl` for large datasets.

**Random colors with no labels.** Don't assign arbitrary colors when there are no labels — use PC-RGB instead (PC1→R, PC2→G, PC3→B, each rescaled to [0,1]). Random colors imply groups that don't exist; PC-RGB is honest about where the color comes from.

**Missing variance bar.** Always show it when the method defines one.

## What it can and cannot say

It can say: "these two classes separate clearly along PC1", "this cluster is mostly type X", "this outlier is the sentence `…`".

It cannot say: "PC1 represents color" or "the model encodes sentiment here" — those need probes and causal tests, not a scatter plot. Geometry alone is correlation, not mechanism.
