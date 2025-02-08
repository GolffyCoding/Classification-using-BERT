from datasets import Dataset, load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# โหลดข้อมูลจาก dataset ของ Pythainlp
ds = load_dataset("pythainlp/thai-financial-dataset")

# ดึงข้อความจากแต่ละ obj และแปลง label เป็นตัวเลข
texts = []
labels = []
label_dict = {}  # ใช้เก็บ mapping ของ label เป็นตัวเลข

for item in ds['train']:  # ใช้ 'train' หรือ 'test' ตามที่ต้องการ
    text = item['text']

    if text:
        parts = text.split('\n', 1)  # แยกข้อความที่ใช้ \n
        if len(parts) > 1:
            label = parts[0].strip()
            texts.append(parts[1].strip())
        else:
            label = parts[0].strip()
            texts.append('')

        # แปลง label เป็นตัวเลข
        if label not in label_dict:
            label_dict[label] = len(label_dict)
        labels.append(label_dict[label])
    else:
        labels.append(0)  # กำหนด default label เป็น 0
        texts.append('')

# สร้าง Dataset
dataset = Dataset.from_dict({"text": texts, "labels": labels})

# แบ่งข้อมูลเป็น train และ eval แค่ครั้งเดียว
split_data = dataset.train_test_split(test_size=0.1)
train_dataset = split_data['train']
eval_dataset = split_data['test']

# โหลด Tokenizer ของ BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenize ข้อความ
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# กำหนดโมเดล
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(label_dict))

# ตั้งค่า arguments สำหรับการฝึก
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True
)

# ใช้ Trainer ในการฝึกโมเดล
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# เริ่มการฝึก
trainer.train()

# ประเมินผลโมเดล
eval_results = trainer.evaluate()
print(eval_results)

# ทำนายผลลัพธ์จากข้อความใหม่
new_text = "ตลาดหุ้นไทยปรับตัวขึ้น 1.2% วันนี้"
inputs = tokenizer(new_text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
inputs = {key: val.to(model.device) for key, val in inputs.items()}  # ส่งไปยัง GPU ถ้ามี
model.eval()

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()

# แสดงผลลัพธ์
predicted_label = [key for key, value in label_dict.items() if value == predictions][0]
print(f"Predicted Label: {predicted_label}")

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
