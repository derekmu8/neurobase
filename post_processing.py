from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional


def debounce_hub_data(
    hub_data: List[Dict[str, List[int]]],
    hub_persistence: int = 3,
    spoke_persistence: int = 5,
    spoke_window: int = 7,
) -> List[Dict[str, List[int]]]:
    """
    Apply simple temporal debouncing to hub/spoke detections.

    Args:
        hub_data: Raw list of {"hub": int, "spokes": list[int]} dictionaries.
        hub_persistence: Number of consecutive frames the same hub must appear
            before being considered valid.
        spoke_persistence: Minimum number of appearances within the recent window
            required before a spoke is shown.
        spoke_window: Size of the sliding window used for spoke persistence.

    Returns:
        A new list of dictionaries with the same structure as `hub_data` but
        with flicker-prone hubs/spokes removed.
    """
    if not hub_data:
        return []

    if hub_persistence < 1:
        raise ValueError("hub_persistence must be >= 1")
    if spoke_persistence < 1:
        raise ValueError("spoke_persistence must be >= 1")

    window = max(spoke_window, spoke_persistence)
    spoke_history: Dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=window))
    smoothed_results: List[Dict[str, List[int]]] = []

    current_hub: Optional[int] = None
    current_run = 0
    last_valid_hub: Optional[int] = None

    for frame in hub_data:
        raw_hub = frame.get("hub")
        raw_spokes = frame.get("spokes", []) or []
        current_spokes = set(raw_spokes)

        # Hub debouncing
        if raw_hub == current_hub:
            current_run += 1
        else:
            current_hub = raw_hub
            current_run = 1

        if current_run >= hub_persistence:
            last_valid_hub = current_hub

        debounced_hub = last_valid_hub

        # Spoke debouncing
        existing_spokes = set(spoke_history.keys())
        for spoke in existing_spokes | current_spokes:
            history = spoke_history[spoke]
            history.append(1 if spoke in current_spokes else 0)

        debounced_spokes = sorted(
            spoke
            for spoke in current_spokes
            if sum(spoke_history[spoke]) >= spoke_persistence
        )

        smoothed_results.append(
            {
                "hub": debounced_hub,
                "spokes": debounced_spokes,
            }
        )

    return smoothed_results

