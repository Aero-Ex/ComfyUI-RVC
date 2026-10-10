import torch


def blend_index_features(feats, index_vectors, index_rate, k=8):
    """FAISS-free replacement for RVC's index retrieval.

    feats: [T, D] float32 source features. index_vectors: [N, D] float32.
    Returns index_rate-blended [T, D]. Matches infer/vc/pipeline.py:186-196,
    whose index is built with faiss.index_factory(d, "IVF,Flat"), i.e. L2
    metric -- faiss reports *squared* L2, so weight = (1/score)^2 is
    (1/dist^2)^2 = 1/dist^4. Exact-zero scores clamp to 1e-8 rather than
    producing inf/NaN.
    """
    dists = torch.cdist(feats.float(), index_vectors.float()).pow(2)
    score, ix = dists.topk(k, largest=False)
    weight = torch.square(1 / score.clamp_min(1e-8))
    weight = weight / weight.sum(dim=1, keepdim=True)
    retrieved = (index_vectors[ix] * weight.unsqueeze(-1)).sum(dim=1)
    return retrieved.to(feats.dtype) * index_rate + feats * (1 - index_rate)


_FAISS_MAGIC = b"IwFl"


def load_index_vectors(path):
    with open(path, "rb") as f:
        if f.read(4) == _FAISS_MAGIC:
            raise RuntimeError(
                "ERROR: %s is a FAISS index, which this pack does not read. "
                "Export it once to .npy, or run without an index." % path)
    if path.lower().endswith(".npy"):
        import numpy as np
        return torch.from_numpy(np.load(path)).float()
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("vectors", "weight", "index_vectors"):
            if key in obj:
                obj = obj[key]
                break
    return torch.as_tensor(obj).float()
