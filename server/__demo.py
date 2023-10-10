import argilla as rg

allowed_fields = (rg.TextField)
allowed_questions = (rg.LabelQuestion, rg.MultiLabelQuestion, rg.RatingQuestion)

for ds in rg.FeedbackDataset.list():
    if len(ds.questions) == len(ds.fields) and len(ds.fields) == 1:
        if isinstance(ds.fields[0], allowed_fields) and isinstance(ds.questions[0], allowed_questions):
            print("yeeha")
