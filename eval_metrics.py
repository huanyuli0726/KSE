from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import numpy as np

# 1. 读取真实标签
true_labels = {}

with open("label1") as f:
    for line in f:
        read_id, _ = line.strip().split()
        true_labels[read_id] = 1

with open("label0") as f:
    for line in f:
        read_id, _ = line.strip().split()
        true_labels[read_id] = 0

# 2. 读取预测结果 + 保存概率
y_true = []
y_pred = []
read_ids = []
probs = []

with open("predictions.txt") as f:
    next(f)  # skip header
    for line in f:
        read_id, pred_label, prob = line.strip().split()
        if read_id in true_labels:
            y_true.append(true_labels[read_id])
            y_pred.append(int(pred_label))
            read_ids.append(read_id)
            probs.append(float(prob))

# 3. 计算指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

specificity = tn / (tn + fp) if (tn + fp) else 0
fpr = fp / (fp + tn) if (fp + tn) else 0
fnr = fn / (fn + tp) if (fn + tp) else 0

# 4. 输出 FP 和 FN 的 read + 概率
with open("fp.txt", "w") as fp_out, open("fn.txt", "w") as fn_out:
    for rid, t, p, prob in zip(read_ids, y_true, y_pred, probs):
        if t == 0 and p == 1:  # False Positive
            fp_out.write(f"{rid}\t{prob:.4f}\n")
        elif t == 1 and p == 0:  # False Negative
            fn_out.write(f"{rid}\t{prob:.4f}\n")

# 5. 输出结果
print("Confusion Matrix:")
print(f"  TP: {tp}")
print(f"  FN: {fn}")
print(f"  TN: {tn}")
print(f"  FP: {fp}")
print("-" * 30)
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"FPR         : {fpr:.4f}")
print(f"FNR         : {fnr:.4f}")

