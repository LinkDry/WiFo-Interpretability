"""Replicate paired first-decoder-layer A/V interventions on QuaDRiGa CDL-C."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from round2_experiment import AV_CONDITIONS, FACTORS, make_probe, run_av_csi
from round2_report import STAT_COLUMNS, ci, effects_by_metric, log_draws, mean_draws
from wifo_pilot import TASKS, atomic_npz, write_table

CDL_HEADS = (0, 1, 2)


def run_cdl(som, output, device):
    source = loadmat(output / 'cdl_c_raw.mat', squeeze_me=True)
    complex_csi = source['H']
    assert complex_csi.shape[1:] == (4, 16, 32, 32)
    assert len(complex_csi) == 128 and np.iscomplexobj(complex_csi) and np.isfinite(complex_csi).all()
    np.testing.assert_allclose(complex_csi[:, [0, 2], 0], complex_csi[:, [1, 3], 0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(complex_csi[:, [0, 1], ..., 0], complex_csi[:, [2, 3], ..., 0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(source['profile_rms_ns'], np.broadcast_to([50., 200.], (128, 2)), atol=1e-8)
    for delta in (1, 2):
        assert not np.allclose(complex_csi[:, 0], complex_csi[:, delta])
    csi = [np.ascontiguousarray(np.stack((data.real, data.imag), axis=1)) for data in complex_csi.transpose(1, 0, 2, 3, 4)]
    reference = make_probe(som, device)
    raw = run_av_csi(reference, csi, batch=8, heads=CDL_HEADS)
    atomic_npz(output / 'av_cdl_c.npz', **raw)
    bootstrap = 118059
    treated = log_draws(raw['errors'], bootstrap)
    baseline = log_draws(raw['baseline'], bootstrap)
    delta = (treated-baseline[:, None, :, None, None, :]).mean(axis=2)
    drift_summary = mean_draws(raw['drift'], bootstrap).mean(axis=2)
    effects, drift = [], []
    for task_index, task in enumerate(TASKS):
        for head_index, head in enumerate(CDL_HEADS):
            name = f'Dec1-H{head + 1}'
            data = delta[task_index, :, head_index]
            comparisons = (*data.transpose(1, 0, 2), data[:, 2]-data[:, 1], data[:, 3]-data[:, 1]-data[:, 2]+data[:, 0])
            labels = (*AV_CONDITIONS, 'V-only minus A-only', 'A x V interaction')
            metrics = (*FACTORS, 'aligned_minus_off_axis')
            for label, values in zip(labels, comparisons):
                for metric, draws in zip(metrics, effects_by_metric(values, task_index)):
                    effects.append(('CDL-C', task, name, label, metric, len(complex_csi), *ci(draws)))
            for metric, draws in zip(metrics, effects_by_metric(drift_summary[task_index, :, head_index], task_index)):
                drift.append(('CDL-C', task, name, metric, len(complex_csi), *ci(draws)))
    write_table(output / 'cdl_av_swap_effects.csv', ['profile', 'task', 'head', 'condition', 'metric', 'n_pairs', *STAT_COLUMNS], effects)
    write_table(output / 'cdl_attn_drift.csv', ['profile', 'task', 'head', 'metric', 'n_pairs', *STAT_COLUMNS], drift)
    timing = {key: np.asarray(source[key]).tolist() for key in ('peak_doppler_hz', 'rms_delay_ns', 'fc', 'velocity',
              'spacing_hz', 'snapshot_s', 'matlab_version', 'quadriga_version')}
    timing.update(n_pairs=len(complex_csi), identity_max_abs_nmse_diff=raw['identity_max_abs_nmse_diff'],
                  baseline_nmse_db_by_task_and_cell=(10*np.log10(raw['baseline'].mean(-1))).tolist(),
                  channel='QuaDRiGa NR-CDL-C. Doppler control is nominal maximum v*fc/c. Delay control is power-weighted nominal PDP RMS.')
    (output / 'cdl_config.json').write_text(json.dumps(timing, indent=2), encoding='utf-8')
    print(f"CDL-C complete. Largest identity NMSE difference {timing['identity_max_abs_nmse_diff']:.3g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--som', type=Path, default=Path('E:/Projects/SoM'))
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/round3_spl_final')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    run_cdl(args.som, args.output, args.device)


if __name__ == '__main__':
    main()
