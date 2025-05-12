import numpy as np
import pysam

BASES = ['A', 'C', 'G', 'T', 'N']
CIGAR_OPS = ['M', 'I', 'D', 'S', 'H', 'N', 'P', '=', 'X']

def one_hot_base(base):
    """将碱基字符转换为 one-hot 编码."""
    vec = [0] * len(BASES)
    if base in BASES:
        vec[BASES.index(base)] = 1
    return vec

def one_hot_cigar(op):
    """将 CIGAR 操作符转换为 one-hot 编码."""
    vec = [0] * len(CIGAR_OPS)
    if op in CIGAR_OPS:
        vec[CIGAR_OPS.index(op)] = 1
    return vec

def get_aligned_features(reference_genome, read, reference_name, max_len=150):
    """从 BAM 读取对齐的特征，并返回 one-hot 编码的特征矩阵."""
    if read.is_unmapped or read.query_sequence is None or read.query_qualities is None:
        feature_dim = 5 + 5 + 1 + 1 + len(CIGAR_OPS)
        return np.zeros((max_len, feature_dim), dtype=np.float32)

    query_seq = read.query_sequence
    base_quals = read.query_qualities
    aligned_pairs = read.get_aligned_pairs(with_seq=True)

    features = []
    feature_dim = 5 + 5 + 1 + 1 + len(CIGAR_OPS)

    for qpos, rpos, ref_base in aligned_pairs:
        if qpos is not None and qpos < len(query_seq):
            qbase = query_seq[qpos]
            qual = base_quals[qpos] / 40.0
        else:
            qbase = 'N'
            qual = 0.0

        if rpos is not None:
            try:
                ref_base = reference_genome.fetch(reference=reference_name, start=rpos, end=rpos+1).upper()
            except Exception:
                ref_base = 'N'
        else:
            ref_base = 'N'

        match_flag = int(qbase == ref_base)

        # Determine CIGAR type for the position
        if qpos is not None and rpos is not None:
            cigar_type = 'M'  # match (aligned position)
        elif rpos is None:
            cigar_type = 'I'  # insertion in the query (query base exists, but no reference base)
        elif qpos is None:
            cigar_type = 'D'  # deletion in the reference (reference base exists, but no query base)
        else:
            cigar_type = 'S'  # soft clipping, a region where sequence is clipped in the read but not in the reference

        # CIGAR can also be 'S', 'H', 'P', 'N', '=', 'X'
        if cigar_type not in CIGAR_OPS:
            cigar_type = 'N'  # Default case for unsupported CIGAR op

        # Create feature vector for this position
        vec = (
            one_hot_base(qbase) +
            one_hot_base(ref_base) +
            [match_flag] +
            [qual] +
            one_hot_cigar(cigar_type)
        )
        features.append(vec)

    # Padding if necessary
    if len(features) >= max_len:
        features = features[:max_len]
    else:
        pad = [[0] * feature_dim] * (max_len - len(features))
        features.extend(pad)

    return np.array(features, dtype=np.float32)

def load_reference_genome(reference_path):
    """加载参考基因组."""
    return pysam.FastaFile(reference_path)

