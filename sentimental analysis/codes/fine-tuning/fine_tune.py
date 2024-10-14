import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset('text', data_files={'train': 'data/train/agnews_corpus_train.txt', 'test': 'data/test/agnews_corpus_test.txt'})

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)  # 4 labels for classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./models',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./models')
tokenizer.save_pretrained('./models')
