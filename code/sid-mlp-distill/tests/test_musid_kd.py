"""CPU-only numerical and gradient regression checks for MuSID-KD objectives."""
import unittest
import torch

from sid_mlp.musid_kd import hcda_loss, rigd_loss, estimate_rigd_scales


class MuSIDLossTests(unittest.TestCase):
    def setUp(self):
        self.codes = torch.tensor([[
            [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0],
            [1, 0, 0, 0], [1, 0, 0, 1], [2, 0, 0, 0],
        ]])
        self.mask = torch.ones(1, 6, dtype=torch.bool)
        self.teacher_steps = torch.tensor([[
            [-0.1, -0.2, -0.3, -0.4], [-0.2, -0.3, -0.4, -0.5],
            [-0.3, -0.4, -0.5, -0.6], [-0.4, -0.5, -0.6, -0.7],
            [-0.5, -0.6, -0.7, -0.8], [-0.6, -0.7, -0.8, -0.9],
        ]], dtype=torch.float32)

    def test_hcda_zero_and_gradient(self):
        s = self.teacher_steps.sum(-1).clone().requires_grad_()
        h = hcda_loss(s, s.detach(), self.codes, self.mask)
        self.assertAlmostEqual(float(h.detach()), 0.0, places=5)
        s2 = (s.detach() * 0.91).requires_grad_()
        h2 = hcda_loss(s2, s.detach(), self.codes, self.mask)
        h2.backward()
        self.assertTrue(torch.isfinite(s2.grad).all())

    def test_hcda_padding_invariance(self):
        a = hcda_loss(self.teacher_steps.sum(-1) * .8,
                      self.teacher_steps.sum(-1), self.codes, self.mask)
        codes = torch.cat([self.codes, torch.full((1, 2, 4), 250, dtype=torch.long)], 1)
        st = torch.cat([self.teacher_steps.sum(-1) * .8, torch.full((1, 2), 999.)], 1)
        tt = torch.cat([self.teacher_steps.sum(-1), torch.full((1, 2), -999.)], 1)
        mask = torch.cat([self.mask, torch.zeros(1, 2, dtype=torch.bool)], 1)
        b = hcda_loss(st, tt, codes, mask)
        self.assertAlmostEqual(float(a), float(b), places=5)

    def test_rigd_mask_and_gradient(self):
        s = (self.teacher_steps + .1 * torch.randn_like(self.teacher_steps)).requires_grad_()
        scale = estimate_rigd_scales(self.codes, self.teacher_steps, torch.tensor([6]),
                                     top_k=2, negative_end=5)
        loss = rigd_loss(self.teacher_steps, s, self.codes, self.mask, scale,
                         top_k=2, negative_end=5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertEqual(float(s.grad[0, :, -1].abs().max()), 0.0)

    def test_empty_eligible(self):
        codes = torch.tensor([[[0,0,0,0],[0,0,0,1],[0,0,0,2],[0,0,0,3]]])
        steps = torch.randn(1, 4, 4, requires_grad=True)
        l = rigd_loss(steps.detach(), steps, codes,
                      torch.ones(1,4,dtype=torch.bool), torch.ones(4),
                      top_k=2, negative_end=4)
        self.assertEqual(float(l), 0.0)


if __name__ == "__main__":
    unittest.main()
