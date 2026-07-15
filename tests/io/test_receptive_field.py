"""Unit tests for the receptive-field decision-map plotter.

Pure numpy/plotly + filesystem work — no GPU, no model load. Synthetic arrays
stand in for the analysis-side grid distributions, point cloud, centroids, and
steering paths. Trace-level assertions use the figure builder (inspecting the
embedded plotly.js bundle would be unreliable — it contains words like
"diamond"/"dash").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from causalab.io.plots.receptive_field import (
    build_receptive_field_figure,
    plot_receptive_field,
)

pytestmark = pytest.mark.unit


def _synthetic(grid_res: int = 5, n_classes: int = 3, n_dims: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    ax = np.linspace(-1.0, 1.0, grid_res)
    mesh = np.meshgrid(*([ax] * n_dims), indexing="ij")
    grid_xy = np.stack([m.reshape(-1) for m in mesh], axis=-1)
    g = grid_xy.shape[0]
    base = np.linspace(-0.8, 0.8, n_classes)
    centroid_xy = np.stack([base] + [np.zeros(n_classes)] * (n_dims - 1), axis=-1)
    d = np.linalg.norm(grid_xy[:, None, :] - centroid_xy[None, :, :], axis=-1)
    grid_argmax = d.argmin(axis=1)
    grid_confidence = rng.uniform(0.2, 1.0, size=g)
    scatter_xy = rng.normal(scale=0.5, size=(20, n_dims))
    scatter_classes = rng.integers(0, n_classes, size=20)

    def _path():
        line = np.linspace(-0.8, 0.8, 10)
        cols = [line] + [np.full(10, 0.1 * k) for k in range(1, n_dims)]
        return np.stack(cols, axis=-1)

    pairs = [(0, n_classes - 1), (0, 1)]  # two pairs -> exercise the path dropdown
    geo = [_path() for _ in pairs]
    lin = [_path() for _ in pairs]
    labels = [str(i) for i in range(n_classes)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:n_classes]
    return dict(
        grid_xy=grid_xy,
        grid_argmax=grid_argmax,
        grid_confidence=grid_confidence,
        grid_res=grid_res,
        axis_ranges=[[-1.0, 1.0]] * n_dims,
        scatter_xy=scatter_xy,
        scatter_classes=scatter_classes,
        centroid_xy=centroid_xy,
        centroid_mask=np.ones(n_classes, dtype=bool),
        geo_paths_xy=geo,
        lin_paths_xy=lin,
        class_labels=labels,
        class_colors=colors,
        pca_components=list(range(n_dims)),
        pair_labels=[f"{labels[a]}->{labels[b]}" for a, b in pairs],
    )


def _field_traces(fig):
    return [t for t in fig.data if getattr(t.marker, "symbol", None) == "square"]


def _menu(fig, last_labels):
    """Return the updatemenu whose final button labels match `last_labels`."""
    for m in fig.layout.updatemenus:
        if [b.label for b in m.buttons][-len(last_labels) :] == list(last_labels):
            return m
    return None


def test_field_per_class_overlays_and_dropdowns():
    fig = build_receptive_field_figure(**_synthetic(n_classes=3))
    # Field is one square-marker trace per class; together they cover the grid.
    fields = _field_traces(fig)
    assert len(fields) == 3
    assert sum(len(t.marker.color) for t in fields) == 5 * 5
    some = next(t for t in fields if len(t.marker.color))
    assert all(c.startswith("rgba(") for c in some.marker.color)

    # Point-cloud dots carry a thin black edge; centroids -> one legend entry.
    clouds = [t for t in fig.data if t.mode == "markers" and t.name in {"0", "1", "2"}]
    assert clouds and all(t.marker.line.color == "black" for t in clouds)
    diamonds = [t for t in fig.data if getattr(t.marker, "symbol", None) == "diamond"]
    assert diamonds and sum(bool(t.showlegend) for t in diamonds) == 1

    # Two pairs -> 4 path lines; default shows only the first pair.
    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == 4 and sum(bool(t.visible) for t in lines) == 2

    # 2-D has a class dropdown + a path dropdown (no slice dropdown).
    assert len(fig.layout.updatemenus) == 2
    class_menu = _menu(fig, ["Class 2"])
    path_menu = _menu(fig, ["All paths", "No paths"])
    assert class_menu is not None and path_menu is not None
    assert [b.label for b in class_menu.buttons][0] == "All classes"
    # Class button restyles field-trace visibility (one class visible).
    vis = class_menu.buttons[1].args[0]["visible"]
    assert sum(vis) == 1 and len(vis) == 3


def test_field_traces_are_leading_indices():
    # The 3-D slice JS (_field_slice_post_script) restyles the per-class field
    # traces by `fieldIdx = range(n_classes)`, which is only correct if those
    # traces occupy the leading positions in fig.data. Pin that invariant: with
    # all overlays present, the square-marker traces must be indices 0..n-1.
    n_classes = 3
    fig = build_receptive_field_figure(**_synthetic(n_dims=3, n_classes=n_classes))
    field_idx = [
        i
        for i, t in enumerate(fig.data)
        if getattr(t.marker, "symbol", None) == "square"
    ]
    assert field_idx == list(range(n_classes))


def test_3d_build_has_per_class_field_and_native_dropdowns():
    fig = build_receptive_field_figure(**_synthetic(grid_res=4, n_classes=3, n_dims=3))
    assert all(isinstance(t, go.Scatter3d) for t in fig.data)
    fields = _field_traces(fig)
    assert len(fields) == 3 and sum(len(t.marker.color) for t in fields) == 4**3
    # Slice orientation/slider live in the JS layer, not native updatemenus:
    # the figure itself carries only the class + path dropdowns.
    assert len(fig.layout.updatemenus) == 2
    assert (
        _menu(fig, ["All classes"]) is None
    )  # "All classes" is the FIRST class button
    assert _menu(fig, ["Class 2"]) is not None
    assert _menu(fig, ["All paths", "No paths"]) is not None
    assert {a.text for a in fig.layout.annotations} >= {"class", "path"}


def test_3d_html_injects_slice_orientation_and_slider(tmp_path: Path):
    out = tmp_path / "rf3d.html"
    plot_receptive_field(output_path=str(out), **_synthetic(grid_res=4, n_dims=3))
    html = out.read_text(encoding="utf-8")
    assert 'id="rf-plot"' in html  # div the JS targets
    assert 'id="rf-orient"' in html  # orientation select
    assert 'id="rf-pos"' in html and 'type="range"' in html  # position slider
    assert "Slice PC0-PC1" in html and "Hide grid" in html
    assert "Plotly.restyle" in html  # recompute hook


def test_2d_html_has_no_slice_slider(tmp_path: Path):
    out = tmp_path / "rf2d.html"
    plot_receptive_field(output_path=str(out), **_synthetic(n_dims=2))
    html = out.read_text(encoding="utf-8")
    assert 'id="rf-orient"' not in html and 'id="rf-pos"' not in html


def test_flat_field_only_has_class_dropdown():
    fig = build_receptive_field_figure(
        encode_confidence=False,
        show_scatter=False,
        show_centroids=False,
        show_paths=False,
        **_synthetic(),
    )
    fields = _field_traces(fig)
    assert len(fig.data) == len(fields)  # only the per-class field traces
    assert all(c.endswith("1.000)") for t in fields for c in t.marker.color)
    assert len(fig.layout.updatemenus) == 1  # class dropdown only (no paths/slice)
    assert _menu(fig, ["All paths", "No paths"]) is None


def test_handles_missing_path_families():
    args = _synthetic()
    args["geo_paths_xy"] = None
    args["lin_paths_xy"] = None
    fig = build_receptive_field_figure(**args)
    assert [t for t in fig.data if t.mode == "lines"] == []
    assert _menu(fig, ["All paths", "No paths"]) is None


def test_writes_html(tmp_path: Path):
    out = tmp_path / "vis" / "receptive_field.html"
    plot_receptive_field(output_path=str(out), **_synthetic())
    assert out.exists()
    assert "Plotly" in out.read_text(encoding="utf-8")
