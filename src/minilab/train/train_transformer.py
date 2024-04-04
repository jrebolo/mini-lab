from config_train import config_en_to_jp as config
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from minilab.models.transformer import TransformerModel

class TextDataset(Dataset):
    def __init__(self, en_sentences, jp_sentences):
        self.en_sentences = en_sentences
        self.jp_sentences = jp_sentences
        
    def __len__(self):
        return len(self.en_sentences)
    def __getitem__(self, idx):
        return self.en_sentences[idx], self.jp_sentences[idx]

def get_dataset(data_path):
    # get data
    with open(f"{data_path}/en_sentences.txt", "r", encoding="utf-8") as f:
        en_sentences = [sentence.strip() for sentence in f.readlines()]
    with open(f"{data_path}/jp_sentences.txt", "r", encoding="utf-8") as f:
        jp_sentences = [sentence.strip() for sentence in f.readlines()]
    
    # create dataset
    dataset = TextDataset(en_sentences, jp_sentences)
    
    return dataset

def get_vocab(dataset):
    en_vocab = set()
    jp_vocab = set()
    
    for en_sentence, jp_sentence in dataset:
        en_vocab.update(en_sentence)
        jp_vocab.update(jp_sentence)
    
    return list(en_vocab), list(jp_vocab)

def get_train_loader(data_path, batch_size):
    
    # get dataset
    train_dataset = get_dataset(data_path)
    
    # get dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader


def train_loop(num_pochs, train_loader, model, optimizer, criterion, device):
    
    model.train()

    for epoch in range(num_pochs):
        print(f"Epoch: {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            
            optimizer.zero_grad()
            # Forward pass

if __name__ == "__main__":
    
    # get config
    batch_size = config["batch_size"]
    data_path = config["data_path"]

    # get dataset
    dataset = get_dataset(data_path)

    en_vocab, jp_vocab = get_vocab(dataset)
    en_vocab = config["TOKENS"] + en_vocab
    jp_vocab = config["TOKENS"] + jp_vocab

    index_to_jp = {idx: token for idx, token in enumerate(jp_vocab)}
    jp_to_index = {token: idx for idx, token in enumerate(jp_vocab)}

    index_to_en = {idx: token for idx, token in enumerate(en_vocab)}
    en_to_index = {token: idx for idx, token in enumerate(en_vocab)}

    train_loader = get_train_loader(data_path, batch_size)
    
    # Init model
    model = TransformerModel(
        input_vocab_size=len(en_vocab),
        output_vocab_size=len(jp_vocab),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=config["max_len"],
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=jp_to_index["<PAD>"], reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    device = torch.device("metal") if torch.cuda.is_available() else torch.device("cpu")

    for params in model.parameters():
        if params.dim() > 1:
            nn.init.xavier_uniform_(params)

    model.to(device)
    model.train()

    for epoch in range(config["num_epochs"]):
        print(f"Epoch: {epoch}")
        for batch_idx, batch in enumerate(train_loader):
            en_batch, jp_batch = batch

            optimizer.zero_grad()
            print(f"En batch: {en_batch}")
            predictions = model(en_batch, jp_batch)
            predictions = predictions.view(-1, predictions.shape[2])
            labels = jp_batch.view(-1)
            valid_indicies = torch.where(labels.view(-1) == jp_to_index["<PAD>]"], False, True)
            loss = loss.sum() / valid_indicies.sum()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")
            print(f"Valid indicies: {valid_indicies.sum()}")
            print(f"Predictions: {predictions.shape}")
            print(f"Labels: {labels.shape}")
            exit(0)