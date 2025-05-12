import pysam
import argparse
import csv
from collections import defaultdict

def load_predictions(pred_file):
    """
    读取 TSV 格式的预测结果，聚合同一 Read_Name 的 label。
    如果同一个 read 多次出现（paired-end），只要有一个是 1，就归为 1。
    """
    label_dict = defaultdict(list)
    with open(pred_file, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            read_name = row['Read_Name']
            label = int(row['Predicted_Label'])
            label_dict[read_name].append(label)

    final_labels = {}
    for read_name, labels in label_dict.items():
        # 只要有一个是 1，就归为 1
        final_labels[read_name] = 1 if 1 in labels else 0

    return final_labels

def split_bam_by_prediction(bam_file, pred_file, out_bam_0, out_bam_1):
    predictions = load_predictions(pred_file)

    in_bam = pysam.AlignmentFile(bam_file, "rb")
    out_0 = pysam.AlignmentFile(out_bam_0, "wb", template=in_bam)
    out_1 = pysam.AlignmentFile(out_bam_1, "wb", template=in_bam)

    count = defaultdict(int)
    total = 0

    for read in in_bam.fetch(until_eof=True):
        name = read.query_name
        if name not in predictions:
            continue
        label = predictions[name]
        if label == 0:
            out_0.write(read)
            count[0] += 1
        elif label == 1:
            out_1.write(read)
            count[1] += 1
        total += 1

    in_bam.close()
    out_0.close()
    out_1.close()

    print(f"共处理 {total} 条 reads：")
    print(f"  写入 {out_bam_0}（label=0）：{count[0]} 条")
    print(f"  写入 {out_bam_1}（label=1）：{count[1]} 条")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据预测结果拆分 BAM 文件（归 1 类优先）")
    parser.add_argument('--bam', required=True, help="输入 BAM 文件")
    parser.add_argument('--pred', required=True, help="预测结果 TSV 文件（含 Read_Name、Predicted_Label）")
    parser.add_argument('--out0', required=True, help="输出 BAM（label=0）")
    parser.add_argument('--out1', required=True, help="输出 BAM（label=1）")
    args = parser.parse_args()

    split_bam_by_prediction(
        bam_file=args.bam,
        pred_file=args.pred,
        out_bam_0=args.out0,
        out_bam_1=args.out1
    )

