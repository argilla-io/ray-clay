
import warnings
from typing import List, Union

from ray import serve
from setfit import SetFitModel

my_deployments = []

@serve.deployment
class SetFitPredictor:
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

# add trainingtask info
def add_deployment(model, workspace, dataset, timestamp):
    deployment = SetFitPredictor.bind(model)
    handle = serve.run(deployment)
    my_deployments.append(
        {
            "workspace": workspace,
            "dataset": dataset,
            "deployment": deployment,
            "handle": handle,
            "timestamp": str(timestamp)
        }
    )

def call_deployment(*args, **kwargs):
    pass

def delete_deployment(model, workspace, dataset, timestamp):
    warnings.warn("Not implemented yet.")
