import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, AutoTokenizer, AdamW

from custom_dataset import IMDbDataset
from model import HuggingFaceCustomClassifier
from utils import read_imdb_split

# get data from files
train_texts, train_labels = read_imdb_split('data/aclImdb/train')
test_texts, test_labels = read_imdb_split('data/aclImdb/test')

# raw data only has train and test sets, let's split the train data to get a validation set
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# import BERT-base pretrained model
HUGGINGFACE_MODEL_NAME = 'bert-base-uncased'
bert = AutoModel.from_pretrained(HUGGINGFACE_MODEL_NAME)
# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAME)
# get config so we can pass it into our custom model and retrieve hidden layers, etc.
config = AutoConfig.from_pretrained(HUGGINGFACE_MODEL_NAME)

# tokenize files first, the tokenizer creates attention masks as well
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=16)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=16)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=16)

# load text into datasets
train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

# instantiate model
num_labels = len(torch.unique(torch.tensor(train_dataset.labels)))
model = HuggingFaceCustomClassifier(bert, config, num_labels)

# instantiate optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# test set size to make sure it runs correctly


optim = AdamW(model.parameters(), lr=5e-5)
cross_entropy = torch.nn.NLLLoss()

for epoch in range(1):
    for batch in train_loader:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        print(input_ids)
        attention_mask = batch['attention_mask'].to(device)
        print(attention_mask)
        labels = batch['labels'].to(device)
        print(labels)

        predicts = model(input_ids, attention_mask)

        loss = cross_entropy(predicts, labels)
        print(loss)
        loss.backward()
        optim.step()

model.eval()
