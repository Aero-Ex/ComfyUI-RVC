import numpy as np
import torch


def blend_index_features(feats, index_vectors, index_rate, k=8, chunk_elements=1 << 26):
    """FAISS-free replacement for RVC's index retrieval.

    feats: [T, D] source features. index_vectors: [N, D] float32 on the same device.
    Returns index_rate-blended [T, D] in feats.dtype. Upstream searches an L2 faiss index,
    which reports *squared* distances, so the weight (1/score)^2 is 1/dist^4. Distances are
    computed in frame chunks so the [T, N] matrix never has to exist at once.
    """
    query = feats.float()
    vectors_sq = index_vectors.square().sum(1)
    rows = max(1, chunk_elements // index_vectors.shape[0])
    retrieved = []
    for chunk in query.split(rows):
        score = torch.addmm(vectors_sq, chunk, index_vectors.t(), alpha=-2).add_(chunk.square().sum(1, keepdim=True))
        score, ix = score.topk(k, largest=False)
        weight = score.clamp_min_(1e-8).reciprocal_().square_()
        weight = weight / weight.sum(dim=1, keepdim=True)
        retrieved.append(torch.einsum("tk,tkd->td", weight, index_vectors[ix]))
    return torch.lerp(query, torch.cat(retrieved), index_rate).to(feats.dtype)


_FAISS_MAGIC = b"IwFl"


def load_index_vectors(path):
    with open(path, "rb") as f:
        if f.read(4) == _FAISS_MAGIC:
            raise RuntimeError(
                "ERROR: %s is a FAISS index, which this pack does not read. "
                "Export it once to .npy, or run without an index." % path)
    if path.lower().endswith(".npy"):
        return torch.from_numpy(np.load(path)).float()
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("vectors", "weight", "index_vectors"):
            if key in obj:
                obj = obj[key]
                break
    return torch.as_tensor(obj).float()
