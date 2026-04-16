from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset, FieldSpec
from inspect_ai.solver import generate
from inspect_ai.scorer import multi_scorer, mean

from eval.scorers import json_validity, render_success, pixel_art_quality


@task
def pixel_art_bench():
    return Task(
        dataset=hf_dataset(
            "AINovice2005/pixel-art-bench",
            split="train",
            sample_fields=FieldSpec(
                input="prompt",
                target="subject",
                id="id",
            ),
        ),
        solver=[generate()],
        scorer=multi_scorer(
            [
                json_validity,
                render_success,
                pixel_art_quality,
            ],
            reducer=mean()
        ),
    )