from transformers import TFAutoModelForSequenceClassification
model = TFAutoModelForSequenceClassification.from_pretrained("./", from_pt=True)
model.save_pretrained("./")