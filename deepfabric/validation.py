import time

from .tui import get_tui


def calculate_expected_paths(mode: str, depth: int, degree: int) -> int:
    """
    Calculate expected number of paths for tree/graph generation.

    Args:
        mode: Generation mode ('tree' or 'graph')
        depth: Depth of the tree/graph
        degree: Branching factor

    Returns:
        Expected number of paths
    """
    if mode == "tree":
        # Tree paths = degree^depth (exact - each leaf is a unique path)
        return degree**depth
    # mode == "graph"
    # Graph paths vary widely due to cross-connections
    # Can range from degree^depth * 0.5 to degree^depth * 2+
    # Use base estimate as rough middle ground, but warn it's approximate
    return degree**depth


def validate_path_requirements(
    mode: str,  # noqa: ARG001 - kept for API compatibility
    depth: int,  # noqa: ARG001 - kept for API compatibility
    degree: int,  # noqa: ARG001 - kept for API compatibility
    num_samples: int | str,  # noqa: ARG001 - kept for API compatibility
    batch_size: int,  # noqa: ARG001 - kept for API compatibility
    loading_existing: bool = False,  # noqa: ARG001 - kept for API compatibility
) -> None:
    """
    Validate topic generation parameters (informational only, no longer errors).

    The generator allocates the exact target across actual unique topics in one pass.

    Args:
        mode: Generation mode ('tree' or 'graph')
        depth: Depth of the tree/graph
        degree: Branching factor
        num_samples: Total samples to generate, or "auto"/percentage string
        batch_size: Batch size for generation (not used in validation, kept for API compat)
        loading_existing: Whether loading existing topic model from file
    """
    # Allocation and empty-topic checks use actual topics after loading/building.
    pass


def show_validation_success(
    mode: str,
    depth: int,
    degree: int,
    num_samples: int | str,
    batch_size: int,  # noqa: ARG001 - kept for API compatibility
    loading_existing: bool = False,
) -> None:
    """
    Show validation success message.

    Args:
        mode: Generation mode ('tree' or 'graph')
        depth: Depth of the tree/graph
        degree: Branching factor
        num_samples: Total samples to generate, or "auto"/percentage string
        batch_size: Batch size for generation (not used in display, kept for API compat)
        loading_existing: Whether loading existing topic model from file
    """
    if loading_existing:
        return

    expected_paths = calculate_expected_paths(mode, depth, degree)
    tui = get_tui()

    # Handle dynamic num_samples (auto or percentage)
    if isinstance(num_samples, str):
        tui.success("Path Validation Passed")
        tui.info(f"  Expected {mode} paths: ~{expected_paths} (depth={depth}, degree={degree})")
        if num_samples == "auto":
            tui.info(
                f"  Requested samples: auto (resolved from actual unique topics; path estimate ~{expected_paths})"
            )
        else:
            # Percentage string like "50%"
            pct = float(num_samples[:-1])
            estimated_samples = max(1, int(expected_paths * pct / 100))
            tui.info(
                f"  Requested samples: {num_samples} (~{estimated_samples} of {expected_paths} paths)"
            )
        if mode == "graph":
            tui.info(
                "  Note: Graph path estimates do not determine unique-topic counts; allocation uses actual non-root nodes"
            )
        print()  # Extra space before topic generation
        time.sleep(0.5)  # Brief pause to allow user to see the information
        return

    tui.success("Path Validation Passed")
    tui.info(f"  Expected {mode} paths: ~{expected_paths} (depth={depth}, degree={degree})")
    tui.info(f"  Requested samples: {num_samples}")

    base, extra = divmod(num_samples, expected_paths)
    tui.info(
        f"  Estimated one-pass allocation: base quota {base}; {extra} topics receive one extra"
    )

    if mode == "graph":
        tui.info(
            "  Note: Graph path estimates do not determine unique-topic counts; allocation uses actual non-root nodes"
        )
    print()  # Extra space before topic generation
    time.sleep(0.5)  # Brief pause to allow user to see the information
