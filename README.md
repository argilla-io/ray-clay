# Commands

## Server
Build
```bash
docker build --pull --rm -f "server.Dockerfile" -t ray-clay-server:latest "."
```
Run locally
```bash
poetry run uvicorn server.main:app
```
## UI

## Todo


Engineering
[ ] create docker.compose
[ ] model store - fsspec - BentoML?

UI:
[ ] basic UI example VUE/Streamlit
[ ] dashboard with running examples
[ ] CRUD model
[ ] CRUD training

Server:
[ ] add ray train
[ ] ray task for suggesstions
[ ] task scheduler - kafka
[ ] add more specific config for models


