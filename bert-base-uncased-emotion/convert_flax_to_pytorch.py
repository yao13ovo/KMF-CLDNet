from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("./", from_flax=True)
model.save_pretrained("./")