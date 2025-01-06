 #Importing the necessary libraries
import json
from collections import Counter
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.nn import CrossEntropyLoss
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#Check if the GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load files
def load_jsonlines(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

# Function to filter sentences where all NER tags are 'O'
def filter_sentences_with_non_zero_tags(data):
    filtered_data = []
    for entry in data:
        if any(tag != 'O' for tag in entry['ner_tags']):  # Check if there is any tag not equal to 'O'
            filtered_data.append(entry)
    return filtered_data

raw_training_data = load_jsonlines('NER-TRAINING.jsonlines')
raw_validation_data = load_jsonlines('NER-VALIDATION.jsonlines')

filtered_training_data = filter_sentences_with_non_zero_tags(raw_training_data)
filtered_validation_data = filter_sentences_with_non_zero_tags(raw_validation_data)

training_data = pd.DataFrame(filtered_training_data)
validation_data = pd.DataFrame(filtered_validation_data)

# Displaying the structure of datasets
print("\nStructure of the Training Data")
training_data.info()

print("\nStructure of the Validation Data")
validation_data.info()

# Function for displaying the distribution of NER tags
def plot_ner_distribution(df, title):
    all_tags = [tag for tags in df['ner_tags'] for tag in tags]
    unique_tags, tag_counts = np.unique(all_tags, return_counts=True)
    tag_distribution = pd.DataFrame({'Tag': unique_tags, 'Count': tag_counts})
    tag_distribution = tag_distribution.sort_values(by='Count', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Tag', y='Count', data=tag_distribution)
    plt.title(f'Distribution of NER classes - {title}')
    plt.xticks(rotation=90)
    plt.show()

# Display of class distribution for the 2 sets
plot_ner_distribution(training_data, 'Filtered Training Dataset')
plot_ner_distribution(validation_data, 'Filtered Validation Dataset')

training_data = training_data.to_dict(orient='records')
validation_data = validation_data.to_dict(orient='records')

# Define unique labels
all_tags = [tag for entry in training_data for tag in entry['ner_tags']]
unique_labels = list(set(all_tags))
print("Unique labels:", unique_labels)

# Calcut class weights
tag_counts = Counter(all_tags)
total_tags = sum(tag_counts.values())
class_weights = [total_tags / tag_counts[label] for label in unique_labels]
class_weights = torch.tensor(class_weights).to(device)

# Preprocessing
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

def encode_data(data, tokenizer, unique_labels, max_length=256):
    input_ids, attention_masks, label_ids = [], [], []
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    for entry in data:
        tokens = entry['tokens']
        tags = entry.get('ner_tags', [])
        
        encodings = tokenizer(tokens, is_split_into_words=True, truncation=True, padding='max_length', max_length=max_length)
        input_ids.append(encodings['input_ids'])
        attention_masks.append(encodings['attention_mask'])
        
        if tags:
            encoded_tags = [label2id.get(tag, -100) for tag in tags]
            padded_tags = encoded_tags[:max_length] + [-100] * (max_length - len(encoded_tags))
            label_ids.append(padded_tags)
        else:
            label_ids.append([-100] * max_length)
    
    return input_ids, attention_masks, label_ids

train_inputs, train_masks, train_labels = encode_data(training_data, tokenizer, unique_labels)
val_inputs, val_masks, val_labels = encode_data(validation_data, tokenizer, unique_labels)

def create_dataloader(input_ids, attention_masks, label_ids, batch_size=4):
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    labels = torch.tensor(label_ids)
    dataset = TensorDataset(inputs, masks, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_inputs, train_masks, train_labels)
val_loader = create_dataloader(val_inputs, val_masks, val_labels)

# Train the model
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss(weight=class_weights)

# Train tqdm
epochs = 20
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    total_loss = 0

    # Integrate tqdm
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}")
    for i, batch in progress_bar:
        input_ids, attention_masks, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits.view(-1, len(unique_labels))
        active_labels = labels.view(-1)
        loss = loss_fn(logits, active_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # Update tqdm
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

# Save the model and the tokenizer
output_dir = 'Data'
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

with open(f"{output_dir}/unique_labels.json", 'w') as f:
    json.dump(unique_labels, f)

print(f"Model and tokenizer saved to {output_dir}")

from sklearn.metrics import classification_report

# Evaluation function
def evaluate_model(model, data_loader, unique_labels):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids, attention_masks, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits

            # Get predictions and active labels
            pred_tags = torch.argmax(logits, dim=2)
            active_labels = labels != -100
            pred_tags = pred_tags[active_labels]
            true_tags = labels[active_labels]
            
            predictions.extend(pred_tags.cpu().numpy())
            true_labels.extend(true_tags.cpu().numpy())
    
    return predictions, true_labels

# Evaluate on validation set
val_predictions, val_true_labels = evaluate_model(model, val_loader, unique_labels)

# Classification report
id2label = {idx: label for idx, label in enumerate(unique_labels)}
val_predictions = [id2label[pred] for pred in val_predictions]
val_true_labels = [id2label[true] for true in val_true_labels]

report = classification_report(val_true_labels, val_predictions, labels=unique_labels, output_dict=False)
print(report)