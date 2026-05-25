from transformers import FlaxAutoModelForSequenceClassification
model = FlaxAutoModelForSequenceClassification.from_pretrained("./", from_pt=True)
model.save_pretrained("./")