# train.py
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from model import CNNModel
from dataset import BAMPairDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bams', nargs='+', required=True)
    parser.add_argument('--refs', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--output_prefix', type=str, default='model')
    return parser.parse_args()

def train(args):
    assert len(args.bams) == len(args.refs) == len(args.labels)

    datasets = [BAMPairDataset(bam, ref, label, args.max_len)
                for bam, ref, label in zip(args.bams, args.refs, args.labels)]
    train_loader = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size, shuffle=True)

    model = CNNModel()
    if args.resume_from and os.path.exists(args.resume_from):
        model.load_state_dict(torch.load(args.resume_from))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), f"{args.output_prefix}_epoch{epoch+1}.pth")

if __name__ == '__main__':
    args = parse_args()
    train(args)

