import glob
from pathlib import Path

import argilla as rg
import ray
from fastapi import FastAPI, HTTPException
from setfit import SetFitModel

from server.managers.ray import add_deployment, my_deployments
from server.managers.thread import DatasetThreader
from server.managers.train import train_textcat_model
from server.models import Model, Payload, Response, TextClassification, Trainer

rg.init(api_key="argilla.apikey", workspace="argilla")
app = FastAPI(
    title="Ray Clay",
    description="Ray Clay is a tool to train and deploy models from Argilla using the Ray framework.",
    docs_url="/docs",
)

@app.on_event("startup")
def start_up():
    global thread
    thread = DatasetThreader()
    thread.start()
    try:
        ds = rg.FeedbackDataset.from_huggingface("argilla/emotion")
        ds.push_to_argilla("emotion")
    except Exception as e:
        print(e)


@app.on_event("shutdown")
def shut_down():
    thread.stop()
    ray.shutdown()

@app.get("/health")
async def health():
    if thread.is_alive():
        return True
    else:
        thread.stop()
        thread.start()
        return False

@app.get("/model/list")
async def model_list():
    return glob.glob("models/*/*/*")

@app.post("/model/train", response_model=Response)
async def train_model(training_data: TextClassification, trainer_data: Trainer, update_config: dict = {}):
    model_path, timestamp = train_textcat_model(
        dataset=trainer_data.dataset,
        workspace=trainer_data.workspace,
        text=training_data.text,
        label=training_data.label,
        label_strategy=training_data.label_strategy,
        trainer_args=trainer_data.dict(exclude={"dataset", "workspace"}),
        config_args=update_config
    )
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

@app.get("/deployment/list")
def deploymentlist():
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

@app.post("/deployment/start", response_model=Response)
async def deploymentstart(data: Model):
    for deployment in my_deployments:
        if data.workspace == deployment["workspace"] and data.dataset == deployment["dataset"] and data.timestamp == deployment["timestamp"]:
            raise HTTPException(status_code=400, detail="Deployment already running")
    else:
        path = Path("models") / Path(data.workspace) / Path(data.dataset) / Path(data.timestamp)
        model = SetFitModel.from_pretrained(path)
        add_deployment(model, data.workspace, data.dataset, data.timestamp)
    return Response(message="Deployment started successfully")

@app.post("/deployment/call")
async def deploymentcall(model: Model, data: Payload):
    for deployment in my_deployments:
        if model.workspace == deployment["workspace"] and model.dataset == deployment["dataset"] and model.timestamp == deployment["timestamp"]:
            remote_results =  deployment["handle"].remote(data.text)
            local_results = ray.get(remote_results)
            return {"values": local_results}
    raise HTTPException(status_code=404, detail="Deployment not found")

@app.post("/deployment/stop", response_model=Response)
async def deploymentstop(data: Model):
    for deployment in my_deployments:
        if deployment["workspace"] == data.workspace and deployment["dataset"] == data.dataset and deployment["timestamp"] == data.timestamp:
            deployment["deployment"].delete()
            return Response(message="Deployment stopped")
    raise HTTPException(status_code=404, detail="Deployment not found")



