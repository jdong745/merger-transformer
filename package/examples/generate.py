"""Generate all modes from the separately supplied inference weights."""
import argparse
import torch
from merger_transformer import WaveformModel, MODES

parser = argparse.ArgumentParser()
parser.add_argument("weights")
parser.add_argument("--device", default="cpu")
parser.add_argument("--output", default="waveform.pt")
args = parser.parse_args()
model = WaveformModel.from_pretrained(args.weights, device=args.device)
x = torch.tensor([2., .2, -.1, .3, -.05, .1, -.2], dtype=torch.float64, device=args.device)
t = torch.linspace(-1100., 80., 2361, dtype=torch.float64, device=args.device)
with torch.no_grad():
    h, psi4 = model.waveforms(x, t)
torch.save({"parameters": x.cpu(), "time_M": t.cpu(), "mode_order": MODES,
            "strain": h.cpu(), "psi4": psi4.cpu()}, args.output)
print(f"Saved {tuple(h.shape)} complex strain and curvature modes to {args.output}")

