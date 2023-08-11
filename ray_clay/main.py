import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List, Union

import argilla as rg
import ray
from argilla.feedback import ArgillaTrainer, FeedbackDataset, TrainingTaskMapping
from fastapi import FastAPI, HTTPException
from ray import serve
from setfit import SetFitModel

from ray_clay.models import Model, Payload, Response, TextClassification, Trainer

rg.init(api_url=os.environ.get("ARGILLA_API_URL_PRE"), api_key=os.environ.get("ARGILLA_API_KEY_PRE"))
app = FastAPI(
    title="Ray Clay",
    description="Ray Clay is a tool to train and deploy models from Argilla using the Ray framework.",
    docs_url="/",
)

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

my_deployments = []

@serve.deployment
class Predictor:
    def __init__(self, model: SetFitModel):
        self.model = model

    def __call__(self, text: Union[str, List[str]]) -> str:
        single = False
        if isinstance(text, str):
            text = [text]
            single = True
        results = self.model.predict_proba(text)
        results = [res.tolist() for res in results]
        if single:
            return results[0]
        return results

def add_deployment(model, workspace, dataset, timestamp):
    deployment = Predictor.bind(model)
    handle = serve.run(deployment)
    my_deployments.append(
        { "workspace": workspace, "dataset": dataset, "deployment": deployment, "handle": handle, "timestamp": str(timestamp)}
    )

@app.get("/health")
async def health():
    return True

@app.get("/model/list")
async def model_list():
    return glob.glob("models/*/*/*")

@app.post("/model/train", response_model=Response)
async def train_model(training_data: TextClassification, trainer_data: Trainer, update_config: dict):
    dataset = FeedbackDataset.from_argilla(
        name=trainer_data.dataset,
        workspace=trainer_data.workspace
    )
    task = TrainingTaskMapping.for_text_classification(
        text=dataset.field_by_name(training_data.text),
        label=dataset.question_by_name(training_data.label),
        label_strategy=training_data.label_strategy
    )
    trainer = ArgillaTrainer(
        task_mapping=task,
        dataset=dataset,
        **trainer_data.dict(exclude={"dataset", "workspace"})
    )
    trainer.update_config(**update_config)

    curr_dt = datetime.now()
    timestamp = int(round(curr_dt.timestamp()))
    model = Path(trainer_data.workspace) / Path(trainer_data.dataset)
    model_path = model_dir / model / str(timestamp)
    trainer.train(model_path)

    add_deployment(model_path, trainer_data.workspace, trainer_data.dataset, str(timestamp))

    return Response(message="Model trained successfully")

@app.post("/model/delete")
def model_delete(data: Model):
    path = Path("models") / Path(data.workspace) / Path(data.dataset) / Path(data.timestamp)
    if path.exists():
        path.unlink()
        return {"message": "Model deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Model not found.")

@app.get("/deploy/list")
def deploy_list():
    # serve.list_deployments()
    if len(my_deployments):
        formatted_deployments = []
        for deployment in my_deployments:
            formatted_deployments.append(
                {
                    "workspace": deployment["workspace"],
                    "dataset": deployment["dataset"],
                    "timestamp": deployment["timestamp"]
                }
            )
        return formatted_deployments
    else:
        raise HTTPException(status_code=404, detail="No deployments found")

@app.post("/deploy/start", response_model=Response)
async def deploy_start(data: Model):
    for deployment in my_deployments:
        if data.workspace == deployment["workspace"] and data.dataset == deployment["dataset"] and data.timestamp == deployment["timestamp"]:
            raise HTTPException(status_code=400, detail="Deployment already running")
    else:
        path = Path("models") / Path(data.workspace) / Path(data.dataset) / Path(data.timestamp)
        model = SetFitModel.from_pretrained(path)
        add_deployment(model, data.workspace, data.dataset, data.timestamp)
    return Response(message="Deployment started successfully")

@app.post("/deploy/call")
async def deploy_call(model: Model, data: Payload):
    for deployment in my_deployments:
        if model.workspace == deployment["workspace"] and model.dataset == deployment["dataset"] and model.timestamp == deployment["timestamp"]:
            remote_results =  deployment["handle"].remote(data.text)
            local_results = ray.get(remote_results)
            return {"values": local_results}
    raise HTTPException(status_code=404, detail="Deployment not found")

@app.post("/deploy/stop", response_model=Response)
async def deploy_stop(data: Model):
    for deployment in my_deployments:
        if deployment["workspace"] == data.workspace and deployment["dataset"] == data.dataset and deployment["timestamp"] == data.timestamp:
            deployment["deployment"].delete()
            return Response(message="Deployment stopped")
    raise HTTPException(status_code=404, detail="Deployment not found")



