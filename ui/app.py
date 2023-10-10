import argilla as rg
import streamlit as st
from argilla.client.feedback.training import (
    ArgillaTrainer,
    TrainingTask,
    TrainingTaskForTextClassification,
)

rg.init(api_key="argilla.apikey", workspace="argilla")

def main():
    st.title("Argilla Trainer")
    tab1, tab2 = st.tabs(["Train", "Deploy"])
    with tab1:
        st.header("Train Models")
        cols = st.columns(2)
        workspaces = [workspace.name for workspace in rg.Workspace.list()]
        ws = cols[0].selectbox("Select Workspace", workspaces)
        datasets = [dataset.name for dataset in rg.FeedbackDataset.list()]
        ds = cols[1].selectbox("Select Dataset", datasets)
        task = cols[0].selectbox("Select Task", ["text-classification", "token-classification"])
        framework = cols[1].selectbox("Select Framework", TrainingTaskForTextClassification._supported_framework_names)
        ds = rg.FeedbackDataset.from_argilla(
            name=ds,
            workspace=ws
        )
        if task == "text-classification":
            fields = [(field.name, str(field.__class__.__name__)) for field in ds.fields if isinstance(field, rg.TextField)]
            questions = [(question.name, str(question.__class__.__name__)) for question in ds.questions if isinstance(question, (rg.LabelQuestion, rg.MultiLabelQuestion))]
            field = cols[0].selectbox("Select Text Field", fields)
            label = cols[1].selectbox("Select Label Field", questions)
            task = TrainingTask.for_text_classification(text=ds.field_by_name(field[0]), label=ds.question_by_name(label[0]))
        else:
            html = (
                """
                <!DOCTYPE html>
                <html>
                <head>
                    <link rel="stylesheet" href="https://pyscript.net/releases/2023.05.1/pyscript.css"/>
                    <script defer src="https://pyscript.net/releases/2023.05.1/pyscript.js"></script>
                    <py-config type="json">
                        {
                            "packages": ["argilla"]
                        }
                    </py-config>
                </head>
                <body>
                <py-repl output-mode="replace">

                </py-repl>
                </body>
                </html>
                """
            )
            st.components.v1.html(html, height=200, scrolling=True)
            st.warning("Not implemented yet")
            st.stop()
        trainer = ArgillaTrainer(task=task, dataset=ds.pull(), framework=framework)
        st.write(trainer)
        if st.button("Train"):
            config = {}
            trainer.update_config(**config)
            trainer.train()
            st.success("Model trained successfully")

    with tab2:
        st.header("Manage Deployments")



if __name__ == "__main__":
    main()

