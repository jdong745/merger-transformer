"""Plot and export a waveform: python -m merger_transformer.plotting --help."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
from .api import WaveformModel
from ._network import MODES
from .projection import project_modes, physical_scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('waveform'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        parser.error("Install the plotting extra: python -m pip install '.[plot]' ("+str(exc)+')')
    config = json.loads(args.config.read_text())
    allowed = {'parameters','time_min_M','time_max_M','dt_M','inclination_rad','azimuth_rad','mass_msun','distance_mpc'}
    if set(config)-allowed:
        parser.error('Unknown config fields: '+', '.join(sorted(set(config)-allowed)))
    params = config['parameters']
    if len(params) != 7 or any(isinstance(v, (list,dict)) for v in params):
        parser.error('parameters must be one seven-vector')
    lo, hi, dt = (float(config.get(k,v)) for k,v in [('time_min_M',-1100),('time_max_M',80),('dt_M',.5)])
    if not all(math.isfinite(v) for v in (lo,hi,dt)) or hi <= lo or dt <= 0:
        parser.error('Require finite time_min_M < time_max_M and dt_M > 0')
    intervals = round((hi-lo)/dt)
    if intervals < 1 or abs(intervals*dt-(hi-lo)) > 1e-9:
        parser.error('Time span must be an integer multiple of dt_M')
    model = WaveformModel.from_pretrained(args.weights, device=args.device)
    t = torch.linspace(lo, hi, intervals+1, dtype=torch.float64, device=args.device)
    x = torch.tensor(params, dtype=torch.float64, device=args.device)
    theta, phi = float(config.get('inclination_rad',math.pi/3)), float(config.get('azimuth_rad',0))
    with torch.no_grad():
        h, p = model.waveforms(x,t)
        hp, pp = project_modes(h,theta,phi), project_modes(p,theta,phi)
    time = t.cpu().numpy()
    h, p, hp, pp = (v.cpu().numpy() for v in (h,p,hp,pp))
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(parameters=np.asarray(params), time_M=time, modes=np.asarray(MODES),
                  strain_modes=h, psi4_modes=p, projected_strain=hp, projected_psi4=pp,
                  inclination_rad=theta, azimuth_rad=phi)
    physical = 'mass_msun' in config or 'distance_mpc' in config
    if physical:
        if not {'mass_msun','distance_mpc'} <= set(config):
            parser.error('Physical plots require both mass_msun and distance_mpc')
        mt, hs, ps = physical_scales(float(config['mass_msun']),float(config['distance_mpc']))
        arrays.update(time_s=time*mt,physical_strain_modes=h*hs,physical_psi4_modes=p*ps,
                      physical_projected_strain=hp*hs,physical_projected_psi4=pp*ps,
                      mass_msun=config['mass_msun'],distance_mpc=config['distance_mpc'])
        plot_t, h,p,hp,pp = time*mt,h*hs,p*ps,hp*hs,pp*ps
        xlabel, hy, py = 'Time (s)', 'Strain', r'$-\ddot h$ (s$^{-2}$)'
    else:
        plot_t=time
        xlabel,hy,py = r'$\tau=t/M$',r'$H=(D/M)h$',r'$P=-\partial_\tau^2 H$'
    np.savez_compressed(str(output)+'.npz',**arrays)
    fig, ax = plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    ax[0].plot(plot_t,hp.real,label=r'$h_+$'); ax[0].plot(plot_t,-hp.imag,label=r'$h_\times$')
    ax[1].plot(plot_t,pp.real,label=r'$\mathrm{Re}\,\Psi_4$'); ax[1].plot(plot_t,-pp.imag,label=r'$-\mathrm{Im}\,\Psi_4$')
    for a in ax: a.legend(loc='upper left'); a.grid(alpha=.2)
    ax[0].set_ylabel(hy); ax[1].set_ylabel(py); ax[1].set_xlabel(xlabel)
    if model.metadata.get('model') == 'private_nr_finetune':
        label = 'Private-NR adaptation, EMA update ' + str(model.metadata['adaptation']['steps'])
    else:
        label = 'v7 constant low-LR STEP048000 EMA'
    fig.suptitle(label + ' — observer projection, no alignment')
    for ext in ('png','pdf'): fig.savefig(str(output)+'_polarizations.'+ext,dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(7,3,figsize=(13,17),sharex=True,layout='constrained')
    for i,((ell,m),a) in enumerate(zip(MODES,ax.flat)):
        a.plot(plot_t,h[i].real,label='Re H'); a.plot(plot_t,h[i].imag,label='Im H')
        a.set_title(f'({ell}, {m:+d})'); a.grid(alpha=.2)
    ax[0,0].legend(loc='upper left')
    for a in ax[-1]: a.set_xlabel(xlabel)
    fig.suptitle('All 21 signed strain modes — '+hy)
    for ext in ('png','pdf'): fig.savefig(str(output)+'_modes.'+ext,dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(7,3,figsize=(13,17),sharex=True,layout='constrained')
    for i,((ell,m),a) in enumerate(zip(MODES,ax.flat)):
        a.plot(plot_t,p[i].real,label='Re P'); a.plot(plot_t,p[i].imag,label='Im P')
        a.set_title(f'({ell}, {m:+d})'); a.grid(alpha=.2)
    ax[0,0].legend(loc='upper left')
    for a in ax[-1]: a.set_xlabel(xlabel)
    fig.suptitle('All 21 signed analytic curvature modes — '+py)
    for ext in ('png','pdf'): fig.savefig(str(output)+'_psi4_modes.'+ext,dpi=160)
    plt.close(fig)
    (Path(str(output)+'_settings.json')).write_text(json.dumps({
        'configuration':config,'weight_metadata':model.metadata,'dtype':'float64',
        'convention':'H=sum H_lm * spin_minus2_Y_lm = h_plus - i h_cross',
        'alignment':'none','curvature':'P=-d2H/dtau2','physical_scaling':physical,
        'mode_order':[list(v) for v in MODES]},indent=2)+'\n')
    print(f'Saved {output}.npz, settings JSON and three PNG/PDF figures.')

if __name__ == '__main__':
    main()
