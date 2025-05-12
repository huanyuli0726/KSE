# predict.py
import argparse
import torch
from torch.utils.data import DataLoader
from model import CNNModel
from dataset import BAMPairDataset
import pandas as pd
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bam', type=str, required=True, help='Path to input BAM file')
    parser.add_argument('--ref', type=str, required=True, help='Path to reference genome')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--output', type=str, default='predictions.txt', help='Path to output prediction file')
    parser.add_argument('--max_len', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model
    model = CNNModel()
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()

    # Dataset
    dataset = BAMPairDataset(args.bam, args.ref, label=0, max_len=args.max_len)  # label is dummy
    loader = DataLoader(dataset, batch_size=args.batch_size)

    read_names = dataset.get_read_names()
    all_predictions = []
    idx = 0

    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            preds = (probs > 0.5).long()
            batch_size = inputs.size(0)
            for i in range(batch_size):
                all_predictions.append({
                    'Index': idx,
                    'Read_Name': read_names[idx],
                    'Predicted_Label': int(preds[i].item()),
                    'Probability': float(probs[i].item())
                })
                idx += 1

    # 合并相同 read name 的预测（若有任意一次预测为 1，则最终为 1）
    grouped = defaultdict(list)
    for item in all_predictions:
        grouped[item['Read_Name']].append(item)

    final_preds = []
    for read_name, entries in grouped.items():
        max_prob = max(e['Probability'] for e in entries)
        any_class1 = any(e['Predicted_Label'] == 1 for e in entries)
        final_preds.append({
            'Read_Name': read_name,
            'Predicted_Label': int(any_class1),
            'Probability': max_prob
        })

    df = pd.DataFrame(final_preds)
    df.to_csv(args.output, sep='\t', index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == '__main__':
    main()

