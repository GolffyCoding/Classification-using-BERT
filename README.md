# Thai Financial Sentiment Classification using BERT

This project implements a Thai financial sentiment classification model using BERT. It utilizes the `pythainlp/thai-financial-dataset` dataset and `bert-base-multilingual-cased` for training a text classification model.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install datasets transformers torch
```

## Dataset

The dataset is loaded from `pythainlp/thai-financial-dataset`. It consists of financial text labeled with sentiment.

## Preprocessing Steps

1. Load the dataset from Hugging Face.
2. Extract text and labels.
3. Convert labels to numerical format.
4. Split dataset into training (90%) and evaluation (10%).
5. Tokenize the text using `BertTokenizer` with `max_length=256`.

## Model Training

1. Load `bert-base-multilingual-cased` with `num_labels` set according to unique labels.
2. Train using `Trainer` from `transformers` with the following settings:
   - `num_train_epochs=3`
   - `per_device_train_batch_size=8`
   - `per_device_eval_batch_size=16`
   - `warmup_steps=500`
   - `weight_decay=0.01`
   - `evaluation_strategy="epoch"`
   - `save_strategy="epoch"`
   - `save_total_limit=2`
   - `load_best_model_at_end=True`

## Model Evaluation

After training, the model is evaluated using the test dataset. Results are printed in the console.

## Making Predictions

To make predictions on new Thai financial text, use:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")

# Sample text
new_text = "ตลาดหุ้นไทยปรับตัวขึ้น 1.2% วันนี้"
inputs = tokenizer(new_text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
inputs = {key: val.to(model.device) for key, val in inputs.items()}

# Make prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()

print(f"Predicted Label: {predictions}")
```

## Saving and Loading Model

After training, the model and tokenizer are saved to `./saved_model`:

```python
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
```

You can reload the model anytime using `from_pretrained('./saved_model')`.

## Conclusion

This project demonstrates how to fine-tune a multilingual BERT model for Thai financial sentiment classification using Hugging Face's `transformers` library. The trained model can be used to analyze financial text sentiment in Thai efficiently.

