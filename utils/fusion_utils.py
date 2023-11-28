
import torch


def l2_norm(inp, axis=1):
    """l2 normalize
    """
    norm = torch.norm(inp, 2, axis, True)
    output = torch.div(inp, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms):
    assert stacked_embeddings.ndim == 3  # (n_features_to_fuse, batch_size, channel)
    assert stacked_norms.ndim == 3  # (n_features_to_fuse, batch_size, 1)

    pre_norm_embeddings = stacked_embeddings * stacked_norms
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)

    return fused, fused_norm
