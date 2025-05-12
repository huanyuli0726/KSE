# dataset.py
import torch
from torch.utils.data import Dataset
import pysam
import numpy as np
from utils import get_aligned_features, load_reference_genome

class BAMPairDataset(Dataset):
    def __init__(self, bam_path, reference_path, label, max_len=150):
        self.bam_path = bam_path
        self.reference = load_reference_genome(reference_path)
        self.label = label
        self.max_len = max_len
        self.reads = []
        self.read_names = []

        bamfile = pysam.AlignmentFile(bam_path, "rb")
        for read in bamfile:
            # Check for YT:Z tag and skip UP-type reads
            yt_tag = read.get_tag('YT') if read.has_tag('YT') else None
            if yt_tag == 'UP':  # Skip UP-type reads
                continue

            if not read.is_unmapped and read.query_sequence:
                self.reads.append(read)
                self.read_names.append(read.query_name)  # 保存 read name
        bamfile.close()

    def __len__(self):
        return len(self.reads)

    def __getitem__(self, idx):
        read = self.reads[idx]
        reference_name = read.reference_name
        features = get_aligned_features(self.reference, read, reference_name, self.max_len)
        label = self.label
        return torch.tensor(features), torch.tensor(label)

    def get_read_names(self):
        return self.read_names

