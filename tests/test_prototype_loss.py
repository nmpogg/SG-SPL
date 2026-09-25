import unittest
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location('sgspl_losses', Path(__file__).resolve().parents[1] / 'src' / 'losses.py')
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)
PrototypeBank = losses.PrototypeBank
prototype_contrastive_loss = losses.prototype_contrastive_loss


class PrototypeContrastiveLossTests(unittest.TestCase):
    def setUp(self):
        self.bank = PrototypeBank(4, 2)
        self.bank.proto_ph[:2] = torch.eye(2)
        self.bank.proto_sk[:2] = torch.eye(2)
        self.bank.proto_mask[:2] = True

    def test_active_classes_and_missing_targets(self):
        sketch = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        photo = torch.randn(3, 2, requires_grad=True)
        labels = torch.tensor([0, 1, 2])

        loss = prototype_contrastive_loss(sketch, photo, labels, self.bank,
                                          temperature=0.1, warmup=2)
        expected = F.cross_entropy(torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
                                   torch.tensor([0, 1]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(sketch.grad[:2].abs().sum().item(), 0)
        self.assertEqual(sketch.grad[2].abs().sum().item(), 0)
        self.assertIsNone(photo.grad)
        self.assertIsNone(self.bank.proto_ph.grad)

    def test_warmup_and_unseen_batch_return_zero(self):
        sketch = torch.randn(1, 2, requires_grad=True)
        photo = torch.randn(1, 2)
        labels = torch.tensor([2])
        self.assertEqual(prototype_contrastive_loss(sketch, photo, labels, self.bank,
                                                    warmup=3).item(), 0)
        self.assertEqual(prototype_contrastive_loss(sketch, photo, labels, self.bank,
                                                    warmup=2).item(), 0)

    def test_optional_reverse_direction(self):
        sketch = torch.tensor([[1.0, 0.0]], requires_grad=True)
        photo = torch.tensor([[0.8, 0.2]], requires_grad=True)
        labels = torch.tensor([0])
        loss = prototype_contrastive_loss(sketch, photo, labels, self.bank,
                                          temperature=0.07, reverse_weight=0.25,
                                          warmup=2)
        loss.backward()
        self.assertGreater(sketch.grad.abs().sum().item(), 0)
        self.assertGreater(photo.grad.abs().sum().item(), 0)
        self.assertIsNone(self.bank.proto_sk.grad)

if __name__ == '__main__':
    unittest.main()
