import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np

# BERT模型路径和tokenizer初始化
bert_name = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)


class MyDataset(Dataset):
    def __init__(self, df):
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def GenerateData(mode):
    train_data_path = 'D:/年报大模型/mynews/data/train.txt'
    dev_data_path = 'D:/年报大模型/mynews/data/dev.txt'
    test_data_path = 'D:/年报大模型/mynews/data/test.txt'

    train_df = pd.read_csv(train_data_path, sep='\t', header=None)
    dev_df = pd.read_csv(dev_data_path, sep='\t', header=None)
    test_df = pd.read_csv(test_data_path, sep='\t', header=None)

    new_columns = ['text', 'label']
    train_df = train_df.rename(columns=dict(zip(train_df.columns, new_columns)))
    dev_df = dev_df.rename(columns=dict(zip(dev_df.columns, new_columns)))
    test_df = test_df.rename(columns=dict(zip(test_df.columns, new_columns)))

    train_dataset = MyDataset(train_df)
    dev_dataset = MyDataset(dev_df)
    test_dataset = MyDataset(test_df)

    if mode == 'train':
        return train_dataset
    elif mode == 'val':
        return dev_dataset
    elif mode == 'test':
        return test_dataset


def save_model(filename):
    torch.save(model.state_dict(), filename)


# 设置训练相关的参数
epochs = 5
batch_size = 16
learning_rate = 1e-5
weight_decay = 1e-4  # 添加权重衰减
clip_norm = 1.0  # 梯度裁剪的最大值
patience = 2  # 早停的容忍次数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = GenerateData(mode='train')
dev_dataset = GenerateData(mode='val')
test_dataset = GenerateData(mode='test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BertClassifier().to(device)

previous_model_path = r"D:\年报大模型\bert\微调第一次\model_epoch_2.pt"
if os.path.exists(previous_model_path):
    print(f"Loading model from {previous_model_path}")
    model.load_state_dict(torch.load(previous_model_path))
else:
    raise FileNotFoundError(f"Model file not found at {previous_model_path}. Please check the path.")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_dev_acc = 0.0
no_improvement_epochs = 0  # 记录连续验证集未提升的epoch次数

for epoch_num in range(epochs):
    model.train()
    total_loss_train = 0
    correct_train = 0
    total_train = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch_num + 1}/{epochs}", unit="batch"):
        input_ids = batch[0]['input_ids'].squeeze(1).to(device)
        attention_mask = batch[0]['attention_mask'].squeeze(1).to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()

        total_loss_train += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    train_acc = correct_train / total_train
    print(f"Epoch [{epoch_num + 1}/{epochs}], Train Loss: {total_loss_train:.4f}, Train Accuracy: {train_acc:.4f}")

    model.eval()
    total_loss_val = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for batch in tqdm(dev_loader, desc=f"Validation {epoch_num + 1}/{epochs}", unit="batch"):
            input_ids = batch[0]['input_ids'].squeeze(1).to(device)
            attention_mask = batch[0]['attention_mask'].squeeze(1).to(device)
            labels = batch[1].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss_val += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    dev_acc = correct_val / total_val
    print(f"Epoch [{epoch_num + 1}/{epochs}], Validation Loss: {total_loss_val:.4f}, Validation Accuracy: {dev_acc:.4f}")

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        no_improvement_epochs = 0
        save_model(f'model_epoch_{epoch_num + 1}_best.pt')
    else:
        no_improvement_epochs += 1

    # 检查早停条件
    if no_improvement_epochs >= patience:
        print("Early stopping triggered. Training terminated.")
        break

model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test", unit="batch"):
        input_ids = batch[0]['input_ids'].squeeze(1).to(device)
        attention_mask = batch[0]['attention_mask'].squeeze(1).to(device)
        labels = batch[1].to(device)

        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)

test_acc = correct_test / total_test
print(f"Test Accuracy: {test_acc:.4f}")
