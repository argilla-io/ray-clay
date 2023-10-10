
from typing import List, Optional, Union

from pydantic import BaseModel


class TextClassification(BaseModel):
    text: str = "tweet"
    label: str = "sentiment"
    label_strategy: Optional[str] = None

class Trainer(BaseModel):
    dataset: str = "covid_tweets_suggestions"
    workspace: str = "recognai"
    framework: str = "setfit"
    model: Optional[str] = None
    train_size: Optional[float] = None
    seed: Optional[int] = None
    gpu_id: Optional[int] = None
    framework_kwargs: Optional[dict] = {}

class Response(BaseModel):
    message: str

class Model(BaseModel):
    dataset: str = "covid_tweets_suggestions"
    workspace: str = "recognai"
    timestamp: str


class Payload(BaseModel):
    text: Union[str, List[str]]