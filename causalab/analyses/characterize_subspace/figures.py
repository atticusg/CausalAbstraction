"""Interactive HTML figure for the characterize_subspace bundle.

:func:`write_projection_explorer_html` is a single linked app over the same
per-document peak-norm representation the judge sees (``peak_value`` =
``‖peak_kdim‖₂``, ``PeakRecord.window_text``, plus the ``(N, k)`` peak-token
vectors for PCA):

- A clickable histogram of each doc's peak-token subspace-activation norm (top).
- Clicking a bin simultaneously (a) lists *all* of that bin's marked context
  windows in a side panel and (b) renders a 3D PCA scatter of the *same*
  documents, coloured by that same norm (explicit colorbar). The bin range,
  panel value, scatter colour, and hover magnitude are one quantity, so the
  list and scatter always show the same set. Hovering a scatter point shows the
  line-wrapped window (peak token bold + underlined) and the norm.

The per-document data is embedded as JSON once and interactivity is driven by a
hand-written ``<script>`` (the embedded-JSON + custom-JS pattern used by
``path_steering/dual_manifold.py``), rather than pre-materialising every view
as Plotly ``updatemenus`` frames.
"""

from __future__ import annotations

import json
import logging
import os

import torch
from torch import Tensor

from causalab.analyses.characterize_subspace.schemas import PeakRecord
from causalab.io.plots.plot_utils import PLOTLY_HTML_CONFIG
from causalab.methods.pca import compute_svd

logger = logging.getLogger(__name__)


_FONT_FAMILY = "Avenir, Avenir Next, Helvetica Neue, sans-serif"


# Shared client helpers: HTML-escape, then turn the ``<<…>>`` peak marker into
# a <mark>. Source ``<<``/``>>`` were sanitised upstream, so the only markers
# are the ones webtext added.
_RENDER_WINDOW_JS = """
  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function renderWindow(w) {
    var e = escapeHtml(w);
    e = e.split('&lt;&lt;').join('<mark>').split('&gt;&gt;').join('</mark>');
    return e;
  }
"""


_EXPLORER_JS = """
  var histDiv = document.getElementById('hist-plot');
  var pcaDiv = document.getElementById('pca-plot');
  var panel = document.getElementById('examples');
  var note = document.getElementById('explorer-note');

  function binIndices(pt) {
    if (pt.pointNumbers && pt.pointNumbers.length !== undefined) {
      return pt.pointNumbers.slice();
    }
    var lo = pt.x - BINSIZE / 2, hi = pt.x + BINSIZE / 2, out = [];
    for (var j = 0; j < DOCS.length; j++) {
      if (DOCS[j].v >= lo - 1e-9 && DOCS[j].v <= hi + 1e-9) out.push(j);
    }
    return out;
  }

  function renderPanel(idxs, lo, hi) {
    // List every document in the bin (the same set shown in the scatter),
    // ranked by subspace-activation norm, strongest first.
    var sorted = idxs.slice().sort(function(a, b) { return DOCS[b].v - DOCS[a].v; });
    var html = '<h3>‖activation‖ in [' + lo.toFixed(3) + ', ' + hi.toFixed(3) +
               ') &mdash; ' + sorted.length + ' doc(s)</h3>';
    if (sorted.length === 0) {
      html += '<p class="empty">(no documents in this bin)</p>';
    } else {
      for (var i = 0; i < sorted.length; i++) {
        var d = DOCS[sorted[i]];
        html += '<div class="ex"><span class="val">' + d.v.toFixed(4) +
                '</span><span class="win">' + renderWindow(d.w) + '</span></div>';
      }
    }
    panel.innerHTML = html;
  }

  function renderScatter(idxs) {
    var xs = [], ys = [], zs = [], col = [], cd = [];
    for (var k = 0; k < idxs.length; k++) {
      var d = DOCS[idxs[k]];
      xs.push(d.c[0]); ys.push(d.c[1]); zs.push(d.c[2]);
      col.push(d.m); cd.push([d.h, d.m]);
    }
    Plotly.react(pcaDiv, [{
      type: 'scatter3d', mode: 'markers',
      x: xs, y: ys, z: zs, customdata: cd,
      hovertemplate:
        '%{customdata[0]}<br><br>' +
        '<b>‖activation‖ in subspace</b> = %{customdata[1]:.4f}<extra></extra>',
      marker: {
        size: 3, color: col, colorscale: 'Viridis', opacity: 0.85,
        cmin: MAG_MIN, cmax: MAG_MAX, showscale: true,
        colorbar: { title: { text: '‖activation‖<br>in subspace', side: 'right' },
                    thickness: 14, len: 0.75 }
      }
    }], pcaDiv.layout);
  }

  function showBin(ev) {
    if (!ev || !ev.points || !ev.points.length) return;
    var pt = ev.points[0];
    var idxs = binIndices(pt);
    var lo = pt.x - BINSIZE / 2, hi = pt.x + BINSIZE / 2;
    renderPanel(idxs, lo, hi);
    renderScatter(idxs);
    note.textContent = '‖activation‖ ∈ [' + lo.toFixed(3) + ', ' + hi.toFixed(3) +
                       '): ' + idxs.length +
                       ' docs — same set in the list and the scatter.';
  }

  histDiv.on('plotly_click', showBin);
"""


def _page(*, title: str, body: str, script_data: str, static_js: str) -> str:
    """Assemble a standalone HTML page (data JSON + static JS, no f-string braces)."""
    return (
        '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        + "<title>"
        + title
        + "</title>\n<style>\n"
        + "  * { box-sizing: border-box; }\n"
        + "  body { margin: 0; padding: 8px 12px; background: white; font-family: "
        + _FONT_FAMILY
        + "; }\n"
        + "  .controls { display: flex; gap: 16px; align-items: center;"
        + " padding: 6px 0; }\n"
        + "  select { font-family: inherit; font-size: 14px; padding: 4px 10px;"
        + " border: 1px solid #ccc; border-radius: 4px; background: white;"
        + " cursor: pointer; }\n"
        + "  #explorer-note { color: #777; font-size: 13px; }\n"
        + "  #hist-plot { width: 100%; height: 300px; }\n"
        + "  .layout { display: flex; gap: 12px; align-items: flex-start; }\n"
        + "  #pca-plot { flex: 1 1 600px; min-width: 480px; height: 560px; }\n"
        + "  #examples { flex: 0 0 380px; max-height: 560px; overflow-y: auto;"
        + " border-left: 1px solid #eee; padding-left: 12px; font-size: 13px; }\n"
        + "  #examples h3 { font-size: 13px; color: #555; font-weight: 600;"
        + " margin: 4px 0 8px; }\n"
        + "  #examples .empty { color: #999; }\n"
        + "  .ex { padding: 6px 0; border-bottom: 1px solid #f2f2f2;"
        + " line-height: 1.4; }\n"
        + "  .ex .val { display: inline-block; min-width: 64px; color: #c0392b;"
        + " font-variant-numeric: tabular-nums; }\n"
        + "  .ex .win mark { background: #ffe08a; padding: 0 1px; }\n"
        + "</style>\n</head>\n<body>\n"
        + body
        + "\n<script>\n(function() {\n"
        + script_data
        + static_js
        + "\n})();\n</script>\n</body>\n</html>\n"
    )


def write_projection_explorer_html(
    out_path: str,
    *,
    peak_kdim: Tensor,
    peak_value: Tensor,
    records: list[PeakRecord],
    nbins: int,
) -> None:
    """Linked histogram + 3D-PCA explorer over the peak-norm representation.

    A clickable histogram of each document's peak-token **subspace-activation
    norm** (``peak_value`` = ``‖peak_kdim‖₂``). Clicking a bin lists *every*
    document in it (marked context windows, strongest first) in a side panel
    **and** renders a 3D PCA scatter of the *same* documents, coloured by that
    same norm (explicit colorbar). The bin range, the panel's value column, the
    scatter colour, and the hover magnitude are all the one quantity, so the
    list and the scatter always show the same set of documents. Hover shows the
    line-wrapped window (peak token bold + underlined) and the norm.
    """
    import plotly.graph_objects as go  # type: ignore[import]
    import plotly.io as pio  # type: ignore[import]

    # ``peak_value`` is already the subspace-activation norm ‖peak_kdim‖, so the
    # histogram value, scatter colour and hover magnitude are the same quantity.
    values = [float(v) for v in peak_value.detach().cpu().tolist()]
    coords, _ncomp, axis_labels = _pca_coords(peak_kdim)
    coords_list = coords.tolist()
    mag_min = min(values) if values else 0.0
    mag_max = max(values) if values else 1.0
    docs = [
        {
            "v": float(v),
            "w": rec.window_text,  # raw window; panel renders <<>> as <mark>
            "h": _hover_window(rec.window_text),  # line-wrapped, bold+underline
            "c": coords_list[i],
            "m": float(v),  # == v; colour/hover magnitude
        }
        for i, (v, rec) in enumerate(zip(values, records))
    ]
    if values:
        vmin, vmax = min(values), max(values)
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    binsize = (vmax - vmin) / max(1, nbins)

    hist = go.Figure(
        data=[
            go.Histogram(
                x=values,
                xbins=dict(start=vmin, end=vmax + binsize * 1e-6, size=binsize),
                marker=dict(color="#4c78a8"),
                hovertemplate="‖activation‖ [%{x}]<br>count %{y}<extra></extra>",
            )
        ]
    )
    hist.update_layout(
        title="Webtext peak-token subspace-activation distribution (click a bin)",
        xaxis_title="Peak-token subspace activation ‖·‖₂ (Euclidean norm over k dims)",
        yaxis_title="Document count",
        bargap=0.02,
        font=dict(family=_FONT_FAMILY),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    hist_html = pio.to_html(
        hist,
        full_html=False,
        include_plotlyjs="cdn",  # pyright: ignore[reportArgumentType]
        div_id="hist-plot",
        config=PLOTLY_HTML_CONFIG,
    )

    # Empty scatter carrying only the scene/axis layout; JS fills data on click.
    pca = go.Figure(data=[go.Scatter3d(x=[], y=[], z=[], mode="markers")])
    pca.update_layout(
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
        ),
        font=dict(family=_FONT_FAMILY),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    pca_html = pio.to_html(
        pca,
        full_html=False,
        include_plotlyjs=False,  # plotly.js already loaded by the histogram
        div_id="pca-plot",
        config=PLOTLY_HTML_CONFIG,
    )

    body = (
        '<div class="controls"><span id="explorer-note">'
        "Click a histogram bar to explore that bin — its context windows appear "
        "on the right and its documents in the 3D PCA scatter."
        "</span></div>\n"
        + hist_html
        + '\n<div class="layout">\n'
        + pca_html
        + '\n<div id="examples"><h3>Click a histogram bar</h3>'
        + '<p class="empty">to list its context windows.</p></div>\n'
        + "</div>"
    )
    script_data = (
        "var DOCS = " + json.dumps(docs) + ";\n"
        "var BINSIZE = " + repr(binsize) + ";\n"
        "var MAG_MIN = " + repr(mag_min) + ";\n"
        "var MAG_MAX = " + repr(mag_max) + ";\n"
    )
    page = _page(
        title="Projection explorer",
        body=body,
        script_data=script_data,
        static_js=_RENDER_WINDOW_JS + _EXPLORER_JS,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(page)


def _pca_coords(peak_kdim: Tensor) -> tuple[Tensor, int, list[str]]:
    """Project peak-token vectors to (at most) 3 PCA coordinates.

    Returns ``(coords, ncomp, axis_labels)`` where ``coords`` is always
    ``(N, 3)`` (trailing columns zero-padded when fewer than 3 components are
    available), ``ncomp`` is the number of real components, and missing axes
    are labelled ``"PC{i} (n/a)"``. Note ``compute_svd`` caps components at
    ``min(N, k) - 1``, so a full 3-D scatter needs ``k >= 4`` and ``N >= 4``;
    smaller subspaces fall back to a padded/degenerate view (logged).
    """
    kdim = peak_kdim.detach().cpu().float()
    n, k = (int(kdim.shape[0]), int(kdim.shape[1])) if kdim.ndim == 2 else (0, 0)
    axis_labels = ["PC1", "PC2", "PC3"]
    coords = torch.zeros(max(n, 0), 3)
    ncomp = 0
    if n >= 2 and k >= 1:
        svd = compute_svd({"peak": kdim}, n_components=3, preprocess="center")["peak"]
        ncomp = int(svd["n_components"])
        if ncomp > 0:
            proj = (kdim - svd["mean"]) @ svd["rotation"]  # (N, ncomp)
            coords[:, :ncomp] = proj[:, :3]
    if ncomp < 3:
        logger.warning(
            "PCA produced only %d component(s) (n=%d, k=%d); padding missing 3D "
            "axes with zeros.",
            ncomp,
            n,
            k,
        )
        for i in range(ncomp, 3):
            axis_labels[i] = f"PC{i + 1} (n/a)"
    return coords, ncomp, axis_labels


def _hover_window(window_text: str, *, wrap_every: int = 10) -> str:
    """Format a peak-token window for a Plotly hover label.

    HTML-escapes the text, inserts a ``<br>`` line break every ``wrap_every``
    whitespace tokens (so long windows don't run off the tooltip), and renders
    the ``<<…>>`` peak marker bold + underlined. Source ``<<``/``>>`` were
    sanitised upstream, so the only markers are the ones webtext added.
    """
    escaped = (
        window_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    tokens = escaped.split()
    lines = [
        " ".join(tokens[i : i + wrap_every]) for i in range(0, len(tokens), wrap_every)
    ]
    wrapped = "<br>".join(lines)
    return wrapped.replace(
        "&lt;&lt;", '<b><span style="text-decoration:underline">'
    ).replace("&gt;&gt;", "</span></b>")
