import json
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from seqeval.metrics import classification_report

model_dir = 'Data'
model = BertForTokenClassification.from_pretrained(model_dir)
tokenizer = BertTokenizerFast.from_pretrained(model_dir)

with open(f'{model_dir}/unique_labels.json', 'r') as f:
    unique_labels = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def load_jsonlines(filepath):
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f]

testing_data = load_jsonlines('NER-TESTING.jsonlines')

'''def encode_data(data, tokenizer, max_length=128):
    input_ids, attention_masks = [], []
    tokens_data = []
    for entry in data:
        tokens = entry['tokens']
        encodings = tokenizer(tokens, is_split_into_words=True, truncation=True, padding='max_length', max_length=max_length)
        input_ids.append(encodings['input_ids'])
        attention_masks.append(encodings['attention_mask'])
        tokens_data.append(tokens)
    return input_ids, attention_masks, tokens_data

test_inputs, test_masks, tokens_data = encode_data(testing_data, tokenizer)
test_loader = DataLoader(TensorDataset(torch.tensor(test_inputs), torch.tensor(test_masks)), batch_size=16)
'''

def encode_data(data, tokenizer, max_length=128):
    input_ids, attention_masks, tokens_data, word_ids_data = [], [], [], []
    for entry in data:
        tokens = entry['tokens']
        encodings = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_offsets_mapping=False,
        )
        input_ids.append(encodings['input_ids'])
        attention_masks.append(encodings['attention_mask'])
        tokens_data.append(tokens)
        word_ids_data.append(encodings.word_ids())
    return input_ids, attention_masks, tokens_data, word_ids_data

test_inputs, test_masks, tokens_data, word_ids_data = encode_data(testing_data, tokenizer)
test_loader = DataLoader(TensorDataset(torch.tensor(test_inputs), torch.tensor(test_masks)), batch_size=16)

pred_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_masks = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_masks)
        predictions = torch.argmax(outputs.logits, dim=-1)

        for i, pred_seq in enumerate(predictions):
            word_ids = word_ids_data.pop(0)  # Get word IDs for the current sequence
            pred_seq_labels = []
            for idx, word_id in enumerate(word_ids):
                # Skip special tokens and padding
                if word_id is None:
                    continue
                # Only take the first subword's prediction for each word
                if idx == 0 or word_id != word_ids[idx - 1]:
                    pred_seq_labels.append(unique_labels[pred_seq[idx].item()])
            pred_labels.append(pred_seq_labels)

        '''for i, pred_seq in enumerate(predictions):
            pred_seq_labels = [
                unique_labels[pred.item()]
                for pred, mask in zip(pred_seq, input_ids[i])
                if mask.item() != tokenizer.pad_token_id
            ]
            pred_labels.append(pred_seq_labels)'''
        
            

'''output_data = []
for original_entry, preds in zip(testing_data, pred_labels):
    output_data.append({
        "unique_id": original_entry["unique_id"],
        "tokens": original_entry["tokens"],
        "ner_tags": preds[:len(original_entry["tokens"])] 
    })'''

# Align predictions with original tokens
output_data = []
for original_entry, preds in zip(testing_data, pred_labels):
    if len(preds) != len(original_entry["tokens"]):
        print(f"Warning: Length mismatch for unique_id {original_entry['unique_id']}")
    # Truncate or pad predictions to match the token length
    preds = preds[:len(original_entry["tokens"])] + ["O"] * (len(original_entry["tokens"]) - len(preds))
    output_data.append({
        "unique_id": original_entry["unique_id"],
        "tokens": original_entry["tokens"],
        "ner_tags": preds
    })

output_path = 'NER-PREDICTIONS.jsonlines'
with open(output_path, 'w') as f:
    for entry in output_data:
        f.write(json.dumps(entry) + "\n")

print(f"Predictions saved to {output_path}")

# Verification of lengths
with open(output_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
        tokens = entry["tokens"]
        ner_tags = entry["ner_tags"]
        assert len(tokens) == len(ner_tags), f"Length mismatch for unique_id {entry['unique_id']}"
print("All sequences in the output file have matching lengths.")

