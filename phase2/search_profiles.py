"""
Named search profiles for constructive benchmark sweeps.

These are intentionally conservative. For larger cubes, the current bottleneck
is often Python-side search expansion rather than raw neural throughput.
"""


SEARCH_PROFILES = {
    "default_like": {
        "beam_width": 32,
        "max_candidates_per_state": 50,
        "placement_ranker": "policy",
        "enable_pocket_pruning": True,
        "max_children_per_layer": None,
        "beam_diversity_slots": 0,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "narrow_4x4": {
        "beam_width": 8,
        "max_candidates_per_state": 12,
        "placement_ranker": "policy",
        "enable_pocket_pruning": True,
        "max_children_per_layer": None,
        "beam_diversity_slots": 0,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "balanced_4x4": {
        "beam_width": 12,
        "max_candidates_per_state": 16,
        "placement_ranker": "policy",
        "enable_pocket_pruning": True,
        "max_children_per_layer": None,
        "beam_diversity_slots": 0,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "narrow_5x5": {
        "beam_width": 10,
        "max_candidates_per_state": 8,
        "placement_ranker": "policy",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 96,
        "beam_diversity_slots": 2,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "balanced_5x5": {
        "beam_width": 12,
        "max_candidates_per_state": 10,
        "placement_ranker": "policy",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 120,
        "beam_diversity_slots": 2,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "layer_capped_5x5": {
        "beam_width": 8,
        "max_candidates_per_state": 12,
        "placement_ranker": "policy",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 72,
        "beam_diversity_slots": 2,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "ultra_capped_5x5": {
        "beam_width": 6,
        "max_candidates_per_state": 8,
        "placement_ranker": "policy",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 48,
        "beam_diversity_slots": 2,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "contact_capped_5x5": {
        "beam_width": 8,
        "max_candidates_per_state": 10,
        "placement_ranker": "contact",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 72,
        "beam_diversity_slots": 3,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "hybrid_capped_5x5": {
        "beam_width": 8,
        "max_candidates_per_state": 10,
        "placement_ranker": "hybrid",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 72,
        "beam_diversity_slots": 2,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "harvest_5x5": {
        "beam_width": 12,
        "max_candidates_per_state": 16,
        "placement_ranker": "contact",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 120,
        "beam_diversity_slots": 4,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    },
    "piece_branch_5x5": {
        "beam_width": 6,
        "max_candidates_per_state": 10,
        "placement_ranker": "policy",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 48,
        "beam_diversity_slots": 3,
        "beam_diversity_metric": "piece_slice_profile",
        "piece_branching_width": 2,
        "piece_branching_slack": 2,
    },
    "harvest_piece_branch_5x5": {
        "beam_width": 12,
        "max_candidates_per_state": 18,
        "placement_ranker": "contact",
        "enable_pocket_pruning": True,
        "max_children_per_layer": 120,
        "beam_diversity_slots": 4,
        "beam_diversity_metric": "piece_slice_profile",
        "piece_branching_width": 2,
        "piece_branching_slack": 2,
    },
}

DEFAULT_PROFILE_BY_GRID = {
    4: "narrow_4x4",
    5: "ultra_capped_5x5",
}

RETRY_PROFILE_BY_GRID = {
    5: "contact_capped_5x5",
}

STRUCTURAL_FALLBACK_BY_GRID = {
    5: {
        "nn_frontier_dfs": True,
        "dfs_timeout": 6.0,
        "dfs_max_frontier_states": 8,
        "dfs_branch_limit": 6,
        "dfs_max_nodes": 25000,
        "dfs_placement_ranker": "contact",
        "dfs_enable_pocket_pruning": True,
        "frontier_portfolio_sources": True,
        "nn_frontier_complete": True,
        "frontier_complete_timeout": 6.0,
        "frontier_complete_max_frontier_states": 4,
        "frontier_complete_max_nodes": 30000,
        "frontier_complete_placement_ranker": "contact",
        "frontier_complete_enable_pocket_pruning": True,
        "frontier_complete_use_transposition": True,
        "frontier_harvest_search_profile": "harvest_5x5",
        "frontier_harvest_timeout": 8.0,
        "frontier_merge_sources": False,
        "frontier_merge_max_states": 10,
        "frontier_merge_per_source_cap": 3,
    },
}


def resolve_search_profile(name):
    """Return a copy of a named search profile."""
    if name not in SEARCH_PROFILES:
        valid = ", ".join(sorted(SEARCH_PROFILES))
        raise ValueError(f"Unknown search profile '{name}'. Valid: {valid}")
    profile = {
        "beam_diversity_slots": 0,
        "beam_diversity_metric": "slice_profile",
        "piece_branching_width": 1,
        "piece_branching_slack": 0,
    }
    profile.update(SEARCH_PROFILES[name])
    return profile


def resolve_runtime_search_settings(
    grid_size,
    *,
    beam_width=None,
    max_candidates_per_state=None,
    placement_ranker=None,
    enable_pocket_pruning=None,
    max_children_per_layer=None,
    beam_diversity_slots=None,
    beam_diversity_metric=None,
    piece_branching_width=None,
    piece_branching_slack=None,
    default_beam_width=32,
    default_max_candidates_per_state=50,
    default_placement_ranker="policy",
    default_enable_pocket_pruning=True,
    default_max_children_per_layer=None,
    default_beam_diversity_slots=0,
    default_beam_diversity_metric="slice_profile",
    default_piece_branching_width=1,
    default_piece_branching_slack=0,
):
    """Fill omitted runtime settings from the grid's default search profile."""
    profile_name = DEFAULT_PROFILE_BY_GRID.get(grid_size)
    profile = resolve_search_profile(profile_name) if profile_name is not None else None

    return {
        "profile_name": profile_name,
        "beam_width": (
            beam_width
            if beam_width is not None
            else (profile["beam_width"] if profile else default_beam_width)
        ),
        "max_candidates_per_state": (
            max_candidates_per_state
            if max_candidates_per_state is not None
            else (
                profile["max_candidates_per_state"]
                if profile else default_max_candidates_per_state
            )
        ),
        "placement_ranker": (
            placement_ranker
            if placement_ranker is not None
            else (profile["placement_ranker"] if profile else default_placement_ranker)
        ),
        "enable_pocket_pruning": (
            enable_pocket_pruning
            if enable_pocket_pruning is not None
            else (
                profile["enable_pocket_pruning"]
                if profile else default_enable_pocket_pruning
            )
        ),
        "max_children_per_layer": (
            max_children_per_layer
            if max_children_per_layer is not None
            else (
                profile["max_children_per_layer"]
                if profile else default_max_children_per_layer
            )
        ),
        "beam_diversity_slots": (
            beam_diversity_slots
            if beam_diversity_slots is not None
            else (
                profile["beam_diversity_slots"]
                if profile else default_beam_diversity_slots
            )
        ),
        "beam_diversity_metric": (
            beam_diversity_metric
            if beam_diversity_metric is not None
            else (
                profile["beam_diversity_metric"]
                if profile else default_beam_diversity_metric
            )
        ),
        "piece_branching_width": (
            piece_branching_width
            if piece_branching_width is not None
            else (
                profile["piece_branching_width"]
                if profile else default_piece_branching_width
            )
        ),
        "piece_branching_slack": (
            piece_branching_slack
            if piece_branching_slack is not None
            else (
                profile["piece_branching_slack"]
                if profile else default_piece_branching_slack
            )
        ),
    }


def resolve_retry_search_settings(
    grid_size,
    *,
    profile_name=None,
):
    """Return a retry profile for a grid when one is configured."""
    chosen = profile_name if profile_name is not None else RETRY_PROFILE_BY_GRID.get(grid_size)
    if chosen is None:
        return None
    out = resolve_search_profile(chosen)
    out["profile_name"] = chosen
    return out


def resolve_structural_fallback_settings(grid_size):
    """Return structural fallback settings for a grid when configured."""
    cfg = STRUCTURAL_FALLBACK_BY_GRID.get(grid_size)
    if cfg is None:
        return None
    return dict(cfg)
