"""Focused correctness checks; no retraining of the full workflow."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
import analysis as a

def run():
    converted=a.innings_to_outs(pd.Series([0,10,10.1,10.2,np.nan]))
    np.testing.assert_allclose(converted,[0,30,31,32,np.nan],equal_nan=True)
    try:
        a.innings_to_outs(pd.Series([10.3]))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid innings accepted')
    s,d=a.load_data(Path('Mid_Major_to_Power_4_Pitcher_Transfers_2024_2026.xlsx'))
    p=a.make_pairs(s,d)
    q=a.make_pairs(s,d,False)
    assert len(p)==70 and p['Player ID'].nunique()==70
    assert (p.outcome_year==p.cohort).all()
    assert (p.source_year<p.cohort).all()
    assert (q.outcome_year!=q.cohort).sum()==3
    assert (p.dest_IP<10).sum()==3
    assert (p.loc[p.cohort<2026,'outcome_year']<2026).all()
    # Predictions cannot depend on destination labels or role information.
    tr,te=p.iloc[:45],p.iloc[45:].copy()
    for name in a.candidates():
        fit=a.fit_named(name,tr,'ERA')
        before=a.predict(fit,te,'ERA')
        altered=te.copy()
        for c in altered:
            if c.startswith('dest_'):altered[c]=99999
        after=a.predict(fit,altered,'ERA')
        np.testing.assert_allclose(before,after)
    o=pd.read_csv('results/nested_cv_predictions.csv')
    assert not o.duplicated(['Player ID','target','model','repeat']).any()
    assert o.groupby(['target','repeat','model']).size().min()>=69
    keys=['Player ID','target','repeat']
    counts=o.groupby(keys).model.nunique()
    assert counts.eq(7).all(), 'Benchmarks must use identical test pitchers'
    np.testing.assert_allclose(o.absolute_error,abs(o.actual-o.predicted))
    ix=pd.read_csv('results/chronological_interval_predictions.csv')
    assert (ix.low<=ix.high).all()
    assert (ix.calibration_n==15).all()
    assert ix.covered.eq((ix.actual>=ix.low)&(ix.actual<=ix.high)).all()
    bundle=joblib.load('results/pitcher_projection_bundle.joblib')
    projection=a.project_pitchers(p.iloc[:2].copy(),bundle)
    assert np.isfinite(projection.filter(regex='^projected_').to_numpy()).all()
    # These are fit checks, never independent accuracy estimates.
    assert projection['projected_K%'].between(0,1).all()
    manifest=json.loads(Path('results/run_manifest.json').read_text())
    assert hashlib.sha256(Path('Mid_Major_to_Power_4_Pitcher_Transfers_2024_2026.xlsx').read_bytes()).hexdigest()==manifest['input_sha256']
    nb=json.loads(Path('power4_pitcher_final_project.ipynb').read_text())
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':compile(''.join(c['source']),f'cell {i}','exec')
    print('Passed: innings, rates, IDs, first-entry pairing, chronology, feature leakage, matched predictions, intervals, saved model, manifest, notebook syntax.')

if __name__=='__main__':run()
