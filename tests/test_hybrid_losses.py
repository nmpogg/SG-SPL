import importlib.util
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


# Import loss functions without requiring Lightning or a downloaded CLIP model.
spec = importlib.util.spec_from_file_location(
    'sgspl_losses', Path(__file__).resolve().parents[1] / 'src' / 'losses.py'
)
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)


class HybridLossTests(unittest.TestCase):
    def setUp(self):
        self.bank = losses.PrototypeBank(4, 2)
        self.bank.proto_ph[0] = torch.tensor([1.0, 0.0])
        self.bank.proto_ph[2] = torch.tensor([0.0, 1.0])
        self.bank.proto_mask[[0, 2, 3]] = True  # class 3 lacks a photo teacher

    def test_prototype_uses_only_valid_active_classes_and_samples(self):
        sketch = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0, 2, 1])
        loss = losses.prototype_contrastive_loss(sketch, labels, self.bank,
                                                 temperature=0.1, warmup=2)
        expected = F.cross_entropy(torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
                                   torch.tensor([0, 1]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(sketch.grad[:2].abs().sum().item(), 0)
        self.assertEqual(sketch.grad[2].abs().sum().item(), 0)
        self.assertIsNone(self.bank.proto_ph.grad)

    def test_warmup_and_unseen_target(self):
        sketch = torch.tensor([[1.0, 0.0]], requires_grad=True)
        self.assertEqual(losses.prototype_contrastive_loss(
            sketch, torch.tensor([0]), self.bank, warmup=3).item(), 0)
        self.assertEqual(losses.prototype_contrastive_loss(
            sketch, torch.tensor([1]), self.bank, warmup=2).item(), 0)

    def test_photo_anchor_only_updates_photo_student(self):
        photo = torch.tensor([[0.8, 0.2]], requires_grad=True)
        anchor = torch.tensor([[0.0, 1.0]], requires_grad=True)
        loss = losses.photo_spherical_loss(photo, anchor)
        expected = 1 - F.cosine_similarity(photo, anchor).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(photo.grad.abs().sum().item(), 0)
        self.assertIsNone(anchor.grad)


if __name__ == '__main__':
    unittest.main()
