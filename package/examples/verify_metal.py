"""Bounded Metal verification; needs actual Apple Silicon for a Metal PASS.

CPU/CUDA options are arithmetic controls, not evidence of Metal execution.
Outputs and synthetic private-NR smoke files are local, never uploaded.
"""
import argparse
import json
import os
import platform
from pathlib import Path
import torch
from merger_transformer import WaveformModel
from merger_transformer._network import transform_inputs, transform_modes


def compare(value, reference):
    v, r = value.detach().cpu().double(), reference.detach().cpu().double()
    return dict(relative_norm=float(torch.linalg.vector_norm(v-r)/torch.linalg.vector_norm(r)),
                maximum_absolute=float((v-r).abs().max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New local directory')
    p.add_argument('--device', choices=('mps', 'cpu', 'cuda'), default='mps')
    p.add_argument('--finetune', action='store_true', help='Also run and resume a synthetic full-network update')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', device=args.device, torch=str(torch.__version__),
                  platform=platform.platform(), machine=platform.machine(),
                  mps_available=torch.backends.mps.is_available(),
                  fallback_env=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
                  mps_fast_math=os.environ.get('PYTORCH_MPS_FAST_MATH'),
                  mps_prefer_metal=os.environ.get('PYTORCH_MPS_PREFER_METAL'),
                  hardware_mps_verified=False)
    path = args.output/'REPORT.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    try:
        if args.device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
            raise RuntimeError('Unset PYTORCH_ENABLE_MPS_FALLBACK for this native MPS verification')
        torch.set_num_threads(1)
        m = WaveformModel.from_pretrained(args.weights, device=args.device, dtype=torch.float32, microbatch_size=1)
        ref = WaveformModel.from_pretrained(args.weights)
        x = torch.tensor([[2., .2, -.1, .3, -.05, .1, -.2],
                          [1., 0., 0., .3, 0., 0., .3],
                          [1.00001, 1e-5, 0., .3, 0., 1e-5, .3],
                          [8., .7, .1, .1, 0., .1, 0.],
                          [20., .2, .1, .6, .1, -.1, .2],
                          [1000., 0., 0., .8, 0., 0., 0.]], dtype=torch.float64)
        # Identical representable inputs/times for all precision comparisons.
        x = x.float().double()
        times = torch.cat((torch.linspace(-1100,80,237,dtype=torch.float64), ref.network.basis.knots[50:380:20]))
        times = times.float().double()
        report['comparisons'] = {}
        with torch.no_grad():
            c = m.coefficients_ri(x)
            cref = ref.coefficients(x)
            for order in range(3):
                value = m.evaluate_coefficients_ri(c,times,derivative=order)
                r = ref.evaluate_coefficients(cref,times,derivative=order)
                r = torch.stack((r.real,r.imag),dim=-1)
                per_row = [compare(v,w) for v,w in zip(value,r)]
                report['comparisons'][f'derivative_{order}'] = per_row
                assert max(z['relative_norm'] for z in per_row) < 5e-4
            batch = m.strain_ri(x[:2],times)
            single = torch.stack([m.strain_ri(row,times) for row in x[:2]])
            report['batch_control'] = compare(batch,single)
            assert report['batch_control']['relative_norm'] < 2e-5
            for g in ('E','P','EP'):
                cg = m.coefficients_ri(transform_inputs(x,g)).cpu()
                complex_c = torch.complex(c.cpu()[...,0],c.cpu()[...,1])
                expected = transform_modes(complex_c,g)
                info = compare(cg,torch.stack((expected.real,expected.imag),dim=-1))
                report[f'symmetry_{g}'] = info
                assert info['relative_norm'] < 2e-5
        xx = x[0].to(device=args.device,dtype=torch.float32).requires_grad_()
        tt = torch.tensor([-500.,-80.,0.,30.],device=args.device,requires_grad=True)
        gradients = torch.autograd.grad(m.strain_ri(xx,tt).square().sum(),(xx,tt))
        xref = xx.detach().cpu().double().requires_grad_()
        tref = tt.detach().cpu().double().requires_grad_()
        references = torch.autograd.grad(ref(xref,tref).abs().square().sum(),(xref,tref))
        report['gradients'] = [compare(a,b) for a,b in zip(gradients,references)]
        assert all(torch.isfinite(v).all() for v in gradients)
        assert max(v['relative_norm'] for v in report['gradients']) < 2e-3
        # Complex convenience output must preserve gradients across explicit CPU copy.
        if args.device == 'mps':
            output = m(xx,tt)
            assert output.device.type == 'cpu' and output.dtype == torch.complex64
            assert all(torch.isfinite(v).all() for v in torch.autograd.grad(output.abs().square().sum(),(xx,tt)))
        if args.finetune:
            from private_nr_smoke import prepare
            from merger_transformer.finetune import parser, run
            manifest = prepare(args.weights,args.output/'synthetic_data')
            argv = ['--weights',str(args.weights),'--manifest',str(manifest),'--output',str(args.output/'run'),
                    '--device',args.device,'--dtype','float32','--trainable','all','--steps','1',
                    '--batch-size','1','--eval-every','1','--replay-per-nr','1',
                    '--replay-train-size','4','--replay-val-size','4']
            run(parser().parse_args(argv))
            state = torch.load(args.output/'run/state.pt',map_location='cpu',weights_only=True)
            parent = torch.load(args.weights,map_location='cpu',weights_only=True)['state_dict']
            assert any(not torch.equal(v,parent[k].float()) for k,v in state['raw'].items()
                       if k.startswith('conditioner.'))
            argv[argv.index('--steps')+1]='2'
            report['synthetic_finetune'] = run(parser().parse_args(argv+['--resume']))
            adapted = WaveformModel.from_pretrained(args.output/'run/last_weights.pt',device=args.device,dtype=torch.float32)
            assert torch.isfinite(adapted.strain_ri(x[:2],times)).all()
        if args.device == 'mps':
            torch.mps.synchronize()
        report.update(status='PASS',hardware_mps_verified=args.device=='mps')
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        path.write_text(json.dumps(report,indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
