from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AdamW
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import datasets
from tqdm import tqdm

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
data_dir = "./dataset"

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df["file_name"][idx]
        text = self.df["text"][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length
        ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100
            for label in labels
        ]
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
        }
        return encoding

print('**BUILDING DATASET**')
# create df for dataset
text_file = os.path.join(data_dir, "latex.txt")
with open(text_file, "r") as f:
    lines = f.readlines()
file_names = []
text_labels = []
for i in range(len(lines)):
    file_name = f"{i}.png"
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        file_names.append(file_name)
        text_labels.append(lines[i].strip())
df = pd.DataFrame({"file_name": file_names, "text": text_labels})
# split up the data into training + testing
train_df, test_df = train_test_split(df, test_size=0.2)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
train_dataset = IAMDataset(root_dir=data_dir+"/", df=train_df, processor=processor)
eval_dataset = IAMDataset(root_dir=data_dir+"/", df=test_df, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)


print('**CONFIGURING MODEL**')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and torch.backends.mps.is_available():
    device = torch.device("mps")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
model.to(device)
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

cer_metric = datasets.load_metric("cer")
def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return cer


print('**TRAINING MODEL**')
optimizer = AdamW(model.parameters(), lr=5e-5)
for epoch in range(1):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))
    model.eval()
    valid_cer = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            outputs = model.generate(batch["pixel_values"].to(device), temperature=0.01, do_sample=True)
            cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
            valid_cer += cer
    print("Validation CER:", valid_cer / len(eval_dataloader))

save_directory = "./my_trained_model"
os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory)


print('**TESTING MODEL**')
import random
finetuned_model = VisionEncoderDecoderModel.from_pretrained("./my_trained_model")
indices = random.sample(range(1000), 20)
images = [(i, Image.open(data_dir + f'/{i}.png').convert("RGB")) for i in indices]
for i, image in images:
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = finetuned_model.generate(pixel_values, temperature=0.01, do_sample=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    with open(data_dir + '/latex.txt', 'r') as file:
        lines = file.readlines()
        line = lines[i].strip()

    print('Prediction for', line, ':  ', generated_text)