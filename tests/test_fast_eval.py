"""Check that batched retrieval keeps the existing metric definitions."""

import importlib.util
from pathlib import Path

import pytest
import torch


spec = importlib.util.spec_from_file_location(
    'retrieval_eval', Path(__file__).resolve().parents[1] / 'src' / 'eval.py'
)
retrieval_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval_eval)


@pytest.mark.parametrize('map_k,prec_k', [(None, 3), (5, 3), (2, None)])
@pytest.mark.parametrize('batch_size', [1, 4, 20])
def test_batched_metrics_match_existing_metrics(map_k, prec_k, batch_size):
    torch.manual_seed(7)
    sk_feats = torch.randn(9, 8)
    ph_feats = torch.randn(17, 8)
    sk_labels = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2])
    ph_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0])

    expected = retrieval_eval.compute_retrieval_metrics(
        sk_feats, ph_feats, sk_labels, ph_labels, map_k=map_k, prec_k=prec_k
    )
    actual = retrieval_eval.compute_retrieval_metrics_batched(
        sk_feats, ph_feats, sk_labels, ph_labels,
        map_k=map_k, prec_k=prec_k, batch_size=batch_size,
    )

    assert actual['map_k'] == expected['map_k']
    assert actual['prec_k'] == expected['prec_k']
    torch.testing.assert_close(actual['mAP'], expected['mAP'], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual['precision'], expected['precision'], atol=1e-6, rtol=1e-6)


def test_batched_metrics_reject_invalid_batch_size():
    with pytest.raises(ValueError, match='batch_size'):
        retrieval_eval.compute_retrieval_metrics_batched(
            torch.empty(1, 2), torch.empty(1, 2),
            torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
            batch_size=0,
        )
