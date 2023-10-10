import logging
import time
from threading import Event, Thread

import argilla as rg

from server._constants import time_interval
from server.managers.ray import call_deployment, my_deployments
from server.managers.train import train_textcat_model


class DatasetThreader(Thread):
    _logger = logging.getLogger("DatasetThreader")
    _logger.setLevel(level=logging.DEBUG)
    allowed_fields = (rg.TextField)
    allowed_questions = (rg.LabelQuestion, rg.MultiLabelQuestion, rg.RatingQuestion)
    stop_flag = Event()

    def _update_existing_deployments(self):
        if my_deployments:
            for deployment in my_deployments:
                self._logger(deployment)
        else:
            self._logger.info("No active deployments")

    def _update_datasets_deployments(self):
        def _check_field_question_match(field, question):
            return isinstance(field, self.allowed_fields) and isinstance(question, self.allowed_questions)

        for ds in rg.FeedbackDataset.list():
            if len(ds.questions) == len(ds.fields) and len(ds.fields) == 1:
                if _check_field_question_match(ds.fields[0], ds.questions[0]):
                    train_textcat_model(
                        dataset=ds.name,
                        workspace=ds.workspace.name,
                        text=ds.fields[0].name,
                        label=ds.questions[0].name,
                    )
                    self._logger.info("I could train a model")
            else:
                for question in ds.questions:
                    for field in ds.fields:
                        if question.name == field.name and _check_field_question_match(field, question):
                            train_textcat_model(
                                dataset=ds.name,
                                workspace=ds.workspace.name,
                                text=ds.field_by_name(field.name),
                                label=ds.question_by_name(question.name),
                            )

    def _add_suggestions_to_deployments(self):
        def add_suggestion_to_deployment(deployment):
            ds = rg.FeedbackDataset.from_argilla(
                name=deployment["name"],
                workspace=deployment["workspace"]
            )
            ds = ds.filter_by(response_status=["draft", "pending"])
            texts = [record.fields[0].value for record in ds if record.metadata.timestamp != deployment["timestamp"]]
            call_deployment(texts)

        [add_suggestion_to_deployment(deployment) for deployment in my_deployments]

    def run(self) -> None:
        while not self.stop_flag.is_set():
            self._update_existing_deployments()
            self._update_datasets_deployments()
            # self.co_add_suggestions_to_deployments()
            time.sleep(time_interval)

    def stop(self) -> None:
        self.stop_flag.set()
        self.join()






