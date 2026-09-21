"""Unit tests for the class-aware cross-modal contrastive loss."""

import importlib.util
from pathlib import Path
import unittest

import torch


LOSSES_PATH = Path(__file__).resolve().parents[1] / 'src' / 'losses.py'
SPEC = importlib.util.spec_from_file_location('sgspl_losses', LOSSES_PATH)
LOSSES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOSSES)
cross_modal_supcon_loss = LOSSES.cross_modal_supcon_loss


class CrossModalSupConLossTest(unittest.TestCase):

    def test_matches_manual_multi_positive_loss(self):
        sk_feat = torch.tensor([
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
        ])
        ph_feat = torch.tensor([
            [0.9, 0.1],
            [1.0, 0.0],
            [0.1, 0.9],
        ])
        cat_idx = torch.tensor([0, 0, 1])
        temperature = 0.1

        loss = cross_modal_supcon_loss(
            sk_feat, ph_feat, cat_idx, temperature=temperature
        )

        sk_n = torch.nn.functional.normalize(sk_feat, dim=-1)
        ph_n = torch.nn.functional.normalize(ph_feat, dim=-1)
        logits = sk_n @ ph_n.t() / temperature
        positives = cat_idx[:, None].eq(cat_idx[None, :]).float()

        sk_log_prob = torch.log_softmax(logits, dim=-1)
        ph_log_prob = torch.log_softmax(logits.t(), dim=-1)
        expected_sk = -(
            (sk_log_prob * positives).sum(-1) / positives.sum(-1)
        ).mean()
        expected_ph = -(
            (ph_log_prob * positives.t()).sum(-1) / positives.t().sum(-1)
        ).mean()
        expected = 0.5 * (expected_sk + expected_ph)

        self.assertTrue(torch.allclose(loss, expected))

    def test_is_finite_and_backpropagates(self):
        sk_feat = torch.randn(8, 16, requires_grad=True)
        ph_feat = torch.randn(8, 16, requires_grad=True)
        cat_idx = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss = cross_modal_supcon_loss(sk_feat, ph_feat, cat_idx)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(sk_feat.grad)
        self.assertIsNotNone(ph_feat.grad)
        self.assertTrue(torch.isfinite(sk_feat.grad).all())
        self.assertTrue(torch.isfinite(ph_feat.grad).all())

    def test_rejects_invalid_temperature(self):
        feats = torch.randn(2, 4)
        labels = torch.tensor([0, 1])

        with self.assertRaises(ValueError):
            cross_modal_supcon_loss(feats, feats, labels, temperature=0.0)


if __name__ == '__main__':
    unittest.main()
