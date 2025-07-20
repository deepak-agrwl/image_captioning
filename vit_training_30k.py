import os
import pandas as pd
import torch
import evaluate
import nltk
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import (
    ViTFeatureExtractor,
    GPT2Tokenizer,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

# === Paths ===
DATA_DIR = "resources/input/flickr30k/flickr30k_images/flickr30k_images"
CSV_PATH = os.path.join("resources/input/flickr30k/flickr30k_images", "results.csv")
OUTPUT_DIR = "vit_gpt2_flickr30k_output"

BATCH_SIZE = 16
EPOCHS = 16
LEARNING_RATE = 5e-5

# === Load CSV ===
df = pd.read_csv(CSV_PATH, sep="|")
df.columns = df.columns.str.strip()
df["image_name"] = df["image_name"].str.strip()
df["comment"] = df["comment"].str.strip()

# === Dataset ===
class Flickr30kDataset(Dataset):
    def __init__(self, dataframe, tokenizer, feature_extractor, image_dir):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row["image_name"])
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        caption = row["comment"]
        tokenized_caption = self.tokenizer(caption, padding="max_length", max_length=64, truncation=True, return_tensors="pt")

        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized_caption.input_ids.squeeze(0),
            "attention_mask": tokenized_caption.attention_mask.squeeze(0),
            "labels": tokenized_caption.input_ids.squeeze(0)
        }

# === Download NLTK ===
nltk.download("punkt")

# === Load Models ===
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224", "gpt2"
)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# === Metrics ===
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
wer = evaluate.load("wer")

def compute_metrics(pred):
    preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

    return {
        "bleu": bleu.compute(predictions=preds, references=labels)["bleu"],
        "rougeL": rouge.compute(predictions=preds, references=labels)["rougeL"],
        "wer": wer.compute(predictions=preds, references=labels),
    }

# === Split Dataset ===
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

train_dataset = Flickr30kDataset(train_df, tokenizer, feature_extractor, DATA_DIR)
val_dataset = Flickr30kDataset(val_df, tokenizer, feature_extractor, DATA_DIR)

# === Training Arguments ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_total_limit=1,
    predict_with_generate=True,
    fp16=False
)

# === Trainer ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator
)

# === Train ===
trainer.train()

# === Save final best model ===
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

# === Plot training & eval loss ===
log_history = trainer.state.log_history
train_loss = [x["loss"] for x in log_history if "loss" in x]
eval_loss = [x["eval_loss"] for x in log_history if "eval_loss" in x]

plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Eval Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Evaluation Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"))
plt.close()
