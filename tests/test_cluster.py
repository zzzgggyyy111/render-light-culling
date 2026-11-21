"""Acceptance tests for the world-space light culling API.

Every test exercises a public stub in ``lightcull.cluster``; in the initial
state all fifteen fail with ``NotImplementedError``.  Tests must not be
modified while fixing the implementation.
"""

import pytest

from lightcull.cluster import (
    LightCullError,
    assign_lights,
    budget_check,
    cluster_grid,
    cull_batch,
    cull_by_frustum,
    estimate_overdraw,
    light_influence,
    merge_clusters,
)


BOUNDS = ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
ALL_X_PLANE = {"planes": [(1.0, 0.0, 0.0, 0.0)]}


def box(lo, hi, cluster_id, pixels=None):
    cluster = {
        "id": cluster_id,
        "min": tuple(float(value) for value in lo),
        "max": tuple(float(value) for value in hi),
    }
    if pixels is not None:
        cluster["pixels"] = float(pixels)
    return cluster


def light(pos, radius=10.0, kind="point"):
    return {
        "pos": tuple(float(value) for value in pos),
        "radius": float(radius),
        "type": kind,
    }


def volume(cluster):
    dx = cluster["max"][0] - cluster["min"][0]
    dy = cluster["max"][1] - cluster["min"][1]
    dz = cluster["max"][2] - cluster["min"][2]
    return dx * dy * dz


def assigned_types(assign):
    return [
        light_item["type"]
        for lights in assign["by_cluster"].values()
        for light_item in lights
    ]


# ---------------------------------------------------------------------------
# One test per documented defect.
# ---------------------------------------------------------------------------


def test_grid_covers_rightmost_boundary_column():
    """Defect 1: the grid must include the clipped final column on every axis."""
    cells = cluster_grid(((0.0, 0.0, 0.0), (5.0, 3.0, 3.0)), 2.0)

    assert len(cells) == 12
    assert min(cell["min"][0] for cell in cells) == pytest.approx(0.0, abs=1e-6)
    assert max(cell["max"][0] for cell in cells) == pytest.approx(5.0, abs=1e-6)
    assert sum(volume(cell) for cell in cells) == pytest.approx(45.0, abs=1e-6)


def test_influence_radius_excludes_far_light():
    """Defect 2: a light beyond its influence radius must contribute nothing."""
    cluster = box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 0)
    near = light((0.0, 0.0, 0.0), 5.0, "near")
    far = light((100.0, 0.0, 0.0), 5.0, "far")

    assert light_influence(near, cluster) == pytest.approx(1.0, abs=1e-6)
    assert light_influence(far, cluster) == pytest.approx(0.0, abs=1e-6)

    result = assign_lights([cluster], [near, far], 4)
    assert [item["type"] for item in result["by_cluster"][0]] == ["near"]
    assert result["overflow"][0] == 0


def test_equal_influence_order_is_stable():
    """Defect 3: equal influences keep a deterministic input order."""
    cluster = box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 0)
    left = light((-1.0, 0.0, 0.0), 5.0, "left")
    right = light((1.0, 0.0, 0.0), 5.0, "right")

    first = assign_lights([cluster], [left, right], 2)
    second = assign_lights([cluster], [left, right], 2)

    assert [item["type"] for item in first["by_cluster"][0]] == ["left", "right"]
    assert [item["type"] for item in second["by_cluster"][0]] == ["left", "right"]


def test_overflow_is_reported_when_cap_truncates():
    """Defect 4: truncation must expose the dropped lights, not hide them."""
    cluster = box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 0)
    lights = [
        light((0.0, 0.0, 0.0), 5.0, "a"),
        light((0.0, 0.0, 0.0), 5.0, "b"),
        light((0.0, 0.0, 0.0), 5.0, "c"),
    ]

    result = assign_lights([cluster], lights, 1)

    assert len(result["by_cluster"][0]) == 1
    assert result["overflow"][0] == 2
    assert [item["type"] for item in result["dropped"][0]] == ["b", "c"]


def test_merge_uses_union_not_intersection():
    """Defect 5: merging adjacent clusters must expand to their union AABB."""
    left = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0)
    right = box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0), 1)

    merged = merge_clusters(left, right)

    assert merged["min"] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert merged["max"] == pytest.approx((2.0, 1.0, 1.0), abs=1e-6)
    assert merged["id"] == 0
    assert volume(merged) == pytest.approx(2.0, abs=1e-6)


def test_frustum_keeps_crossing_cluster():
    """Defect 6: a cluster straddling a frustum plane must survive."""
    crossing = box((-2.0, -1.0, -1.0), (1.0, 1.0, 1.0), 7)
    outside = box((-4.0, -1.0, -1.0), (-3.0, 1.0, 1.0), 8)
    inside = box((1.0, -1.0, -1.0), (2.0, 1.0, 1.0), 9)

    visible = cull_by_frustum([crossing, outside, inside], ALL_X_PLANE)

    assert visible == [7, 9]


def test_budget_counts_pixels_not_light_count():
    """Defect 7: the budget is in light-pixel samples, not a count of lights."""
    wide = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0, pixels=1000.0)
    tiny = box((10.0, 0.0, 0.0), (11.0, 1.0, 1.0), 1, pixels=10.0)
    lights = [
        light((0.5, 0.5, 0.5), 2.0, "wide"),
        light((10.5, 0.5, 0.5), 2.0, "tiny"),
    ]

    assign = assign_lights([wide, tiny], lights, 4)
    issues = budget_check(assign, 500.0)

    # 1000 * 1 + 10 * 1 = 1010 light-pixel samples, but only 2 lights.
    assert any(issue.startswith("over_budget:") for issue in issues)


def test_overdraw_is_pixel_weighted():
    """Defect 8: overdraw must weight light overlap by cluster pixel coverage."""
    wide = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0, pixels=1000.0)
    tiny = box((100.0, 0.0, 0.0), (101.0, 1.0, 1.0), 1, pixels=1.0)
    lights = [
        light((0.5, 0.5, 0.5), 2.0, "wide-0"),
        light((0.5, 0.5, 0.5), 2.0, "wide-1"),
    ]
    lights.extend(
        light((100.5, 0.5, 0.5), 2.0, f"tiny-{index}") for index in range(10)
    )

    assign = assign_lights([wide, tiny], lights, 20)
    overdraw = estimate_overdraw([wide, tiny], assign)

    # extra = (2 - 1) * 1000 + (10 - 1) * 1 = 1009
    # base  = 1000 + 1 = 1001
    assert overdraw == pytest.approx(1009.0 / 2010.0, abs=1e-6)


def test_batch_frames_do_not_share_assignments():
    """Defect 9: per-frame assignment tables must be independent."""
    config = {"bounds": BOUNDS, "size": 2.0, "max_per_cluster": 2}
    frames = [
        {"lights": [light((0.5, 0.5, 0.5), 1.0, "frame0")], "frustum": ALL_X_PLANE},
        {"lights": [light((3.5, 3.5, 3.5), 1.0, "frame1")], "frustum": ALL_X_PLANE},
    ]

    result = cull_batch(frames, config)

    types0 = assigned_types(result["frames"][0]["assign"])
    types1 = assigned_types(result["frames"][1]["assign"])
    assert types0 == ["frame0"]
    assert types1 == ["frame1"]


def test_grid_rejects_non_positive_size():
    """Defect 10: size <= 0 must fail cleanly instead of dividing by zero."""
    with pytest.raises(LightCullError):
        cluster_grid(BOUNDS, 0.0)
    with pytest.raises(LightCullError):
        cluster_grid(BOUNDS, -2.0)


# ---------------------------------------------------------------------------
# Additional contract tests.
# ---------------------------------------------------------------------------


def test_single_light_single_cluster_assignment():
    """A light at the centre of one cluster is assigned with influence 1."""
    cluster = box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), 0)
    only_light = light((1.0, 1.0, 1.0), 10.0, "point")

    assert light_influence(only_light, cluster) == pytest.approx(1.0, abs=1e-6)
    result = assign_lights([cluster], [only_light], 4)

    assert len(result["by_cluster"][0]) == 1
    assert result["by_cluster"][0][0]["type"] == "point"
    assert result["overflow"][0] == 0
    assert result["dropped"][0] == []


def test_zero_cap_yields_empty_assignments_and_reports():
    """A zero per-cluster cap assigns nothing but reports every candidate."""
    cluster = box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0), 0)
    lights = [
        light((1.0, 1.0, 1.0), 5.0, "a"),
        light((1.0, 1.0, 1.0), 5.0, "b"),
    ]

    result = assign_lights([cluster], lights, 0)

    assert result["by_cluster"][0] == []
    assert result["overflow"][0] == 2
    assert [item["type"] for item in result["dropped"][0]] == ["a", "b"]


def test_empty_scene_returns_empty_result():
    """An empty scene yields empty structures without special cases."""
    assert cull_by_frustum([], ALL_X_PLANE) == []

    assignment = assign_lights([], [], 4)
    assert assignment["clusters"] == {}
    assert assignment["by_cluster"] == {}
    assert assignment["overflow"] == {}
    assert assignment["dropped"] == {}

    batched = cull_batch(
        [],
        {"bounds": BOUNDS, "size": 2.0, "max_per_cluster": 4},
    )
    assert batched == {"frame_count": 0, "frames": []}


def test_batch_rejects_zero_size():
    """Batch configuration with size 0 raises LightCullError."""
    frames = [{"lights": [], "frustum": ALL_X_PLANE}]

    with pytest.raises(LightCullError):
        cull_batch(
            frames,
            {"bounds": BOUNDS, "size": 0.0, "max_per_cluster": 4},
        )


def test_batch_success_preserves_frame_order():
    """Successful batches preserve frame order and do not mutate input."""
    config = {"bounds": BOUNDS, "size": 2.0, "max_per_cluster": 2}
    frames = [
        {"lights": [light((1.0, 1.0, 1.0), 0.9, "frame0")], "frustum": ALL_X_PLANE},
        {"lights": [light((3.0, 1.0, 1.0), 0.9, "frame1")], "frustum": ALL_X_PLANE},
        {"lights": [light((3.0, 3.0, 3.0), 0.9, "frame2")], "frustum": ALL_X_PLANE},
    ]
    original = [
        {
            "lights": [dict(light_item) for light_item in frame["lights"]],
            "frustum": {"planes": list(frame["frustum"]["planes"])},
        }
        for frame in frames
    ]

    result = cull_batch(frames, config)

    assert result["frame_count"] == 3
    assert [frame["index"] for frame in result["frames"]] == [0, 1, 2]
    assert [assigned_types(frame["assign"]) for frame in result["frames"]] == [
        ["frame0"],
        ["frame1"],
        ["frame2"],
    ]
    assert frames == original
