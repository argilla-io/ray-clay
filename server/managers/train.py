from datetime import datetime
from pathlib import Path
from typing import Optional

from argilla.feedback import ArgillaTrainer, FeedbackDataset, TrainingTaskMapping

from server._constants import model_dir


def train_textcat_model(
    dataset: str,
    workspace: str,
    text: str,
    label: str,
    label_strategy: Optional[str] = None,
    trainer_args: Optional[dict] = {},
    config_args: Optional[dict] = {}
):
    ds = FeedbackDataset.from_argilla(
        name=dataset,
        workspace=workspace
    )
    ds = ds.pull()
    ds.records = [record for record in ds.records[:30]]

    task = TrainingTaskMapping.for_text_classification(
        text=ds.field_by_name(text),
        label=ds.question_by_name(label),
        label_strategy=label_strategy
    )
    trainer = ArgillaTrainer(
        task=task,
        dataset=ds,
        framework="setfit",
        **trainer_args
    )
    trainer.update_config(**config_args)

    curr_dt = datetime.now()
    timestamp = int(round(curr_dt.timestamp()))
    model = Path(workspace) / Path(dataset)
    model_path = model_dir / model / str(timestamp)
    trainer.train(model_path)

    return model_path, timestamp