"""Unit tests for off-diagonal structural consistency losses."""

import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


LOSSES_PATH = Path(__file__).resolve().parents[1] / 'src' / 'losses.py'
SPEC = importlib.util.spec_from_file_location('sgspl_losses', LOSSES_PATH)
LOSSES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOSSES)


class StructuralLossTest(unittest.TestCase):

    def _inputs(self):
        torch.manual_seed(7)
        n_classes, dim = 5, 8
        sk_raw = torch.randn(n_classes, dim, requires_grad=True)
        ph_raw = torch.randn(n_classes, dim, requires_grad=True)
        sk_feat = F.normalize(sk_raw, dim=-1)
        ph_feat = F.normalize(ph_raw, dim=-1)
        labels = torch.arange(n_classes)
        bank = LOSSES.PrototypeBank(n_classes, dim, momentum=0.9)
        bank.update(sk_feat.detach(), labels, 'sk')
        bank.update(ph_feat.detach(), labels, 'ph')
        text = F.normalize(torch.randn(n_classes, dim), dim=-1)
        anchor = text @ text.t()
        return sk_raw, ph_raw, sk_feat, ph_feat, labels, bank, anchor

    def test_off_diagonal_loss_ignores_anchor_diagonal(self):
        _, _, sk, ph, labels, bank, anchor = self._inputs()
        changed = anchor.clone()
        changed.fill_diagonal_(-20.0)

        for dist in ('mse', 'kl', 'sym_kl', 'js'):
            first = LOSSES.structural_losses(
                sk, ph, labels, bank, anchor, dist=dist, T=0.1,
                warmup=2, exclude_diagonal=True,
            )
            second = LOSSES.structural_losses(
                sk, ph, labels, bank, changed, dist=dist, T=0.1,
                warmup=2, exclude_diagonal=True,
            )
            self.assertTrue(torch.allclose(first[0], second[0]))
            self.assertTrue(torch.allclose(first[1], second[1]))

    def test_xmod_alone_backpropagates_to_both_modalities(self):
        sk_raw, ph_raw, sk, ph, labels, bank, anchor = self._inputs()
        _, loss_xmod = LOSSES.structural_losses(
            sk, ph, labels, bank, anchor, dist='js', T=0.1,
            warmup=2, exclude_diagonal=True,
        )
        loss_xmod.backward()

        self.assertIsNotNone(sk_raw.grad)
        self.assertIsNotNone(ph_raw.grad)
        self.assertGreater(sk_raw.grad.abs().sum().item(), 0.0)
        self.assertGreater(ph_raw.grad.abs().sum().item(), 0.0)

    def test_no_proto_grad_disables_structural_gradients(self):
        _, _, sk, ph, labels, bank, anchor = self._inputs()
        loss_ssc, loss_xmod = LOSSES.structural_losses(
            sk, ph, labels, bank, anchor, dist='kl', T=0.1,
            warmup=2, no_proto_grad=True, exclude_diagonal=True,
        )
        self.assertFalse(loss_ssc.requires_grad)
        self.assertFalse(loss_xmod.requires_grad)

    def test_classification_loss_averages_modalities(self):
        _, _, sk, ph, labels, _, anchor = self._inputs()
        text = F.normalize(torch.randn(anchor.shape[0], sk.shape[1]), dim=-1)
        scale = torch.tensor(10.0)
        actual = LOSSES.classification_loss(sk, ph, labels, text, scale)
        expected = 0.5 * (
            F.cross_entropy(scale * sk @ text.t(), labels)
            + F.cross_entropy(scale * ph @ text.t(), labels)
        )
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
