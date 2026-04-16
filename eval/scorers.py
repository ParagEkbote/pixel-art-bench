from inspect_ai.scorer import scorer, accuracy, mean
import json


@scorer(metrics=[accuracy()])
def json_validity(sample, output, **kwargs):
    output = output.text if hasattr(output, "text") else output

    if not isinstance(output, str):
        return 0.0

    try:
        json.loads(output)
        return 1.0
    except json.JSONDecodeError:
        return 0.0


@scorer(metrics=[accuracy()])
def render_success(sample, output, **kwargs):
    output = output.text if hasattr(output, "text") else output

    if not isinstance(output, str):
        return 0.0

    try:
        data = json.loads(output)
        grid = data.get("grid", [])

        if (
            isinstance(grid, list)
            and len(grid) == 24
            and all(isinstance(row, list) and len(row) == 24 for row in grid)
            and all(isinstance(cell, int) for row in grid for cell in row)
        ):
            return 1.0

        return 0.0

    except (json.JSONDecodeError, TypeError):
        return 0.0


@scorer(metrics=[mean()])
def pixel_art_quality(sample, output, **kwargs):
    output = output.text if hasattr(output, "text") else output

    if not isinstance(output, str):
        return 0.0

    try:
        data = json.loads(output)
        grid = data.get("grid", [])

        if not grid:
            return 0.0

        unique_colors = len(set(cell for row in grid for cell in row))
        return min(unique_colors / 10.0, 1.0)

    except (json.JSONDecodeError, TypeError):
        return 0.0