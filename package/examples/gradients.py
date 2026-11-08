"""Differentiate a strain sample with respect to all seven binary parameters."""
import argparse
import torch
from merger_transformer import WaveformModel, MODES

parser = argparse.ArgumentParser()
parser.add_argument("weights")
args = parser.parse_args()
model = WaveformModel.from_pretrained(args.weights, microbatch_size=1)
x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], dtype=torch.float64, requires_grad=True)
t = torch.tensor([-20.25], dtype=torch.float64, requires_grad=True)
h = model(x, t)
sample = h[MODES.index((2, 2)), 0].real
dx, dt = torch.autograd.grad(sample, (x, t), create_graph=True)
ddt = torch.autograd.grad(dt.sum(), t)[0]
curvature = model.psi4(x.detach(), t.detach())[MODES.index((2, 2)), 0].real
print("d Re(h22) / dx:", dx.detach())
print("d Re(h22) / dtau:", dt.detach())
print("AD second derivative plus analytic Psi4:", (ddt + curvature).detach())
