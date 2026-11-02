import numpy as np
import pandas as pd
import pytest
from src.data import add_training_rul, attach_test_rul, load_subset
from src.features import engineer_features, latest_engine_features, SequentialFeatureEngineer
from src.metrics import regression_metrics, nasa_score

def history():
    rows=[]
    for engine in [1, 2]:
        for cycle in range(1, 5):
            rows.append({"engine_id":engine,"cycle":cycle,"sensor_1":float(engine*10+cycle)})
    return pd.DataFrame(rows)

def test_training_rul_is_engine_local_and_capped():
    result=add_training_rul(history(),cap=2)
    assert result[result.engine_id==1].rul.tolist()==[2,2,1,0]
    assert result[result.engine_id==2].rul.tolist()==[2,2,1,0]

def test_test_labels_add_unseen_remaining_cycles():
    test=history()
    result=attach_test_rul(test,np.array([7.,3.]))
    assert result[result.engine_id==1].rul.tolist()==[10,9,8,7]
    assert result[result.engine_id==2].rul.tolist()==[6,5,4,3]

def test_features_are_ordered_and_latest_matches_full_transform():
    h=history().query("engine_id == 1")
    full=engineer_features(h,["sensor_1"])
    latest=latest_engine_features(h,["sensor_1"])
    assert list(latest.columns)==list(full.columns)
    pd.testing.assert_series_equal(latest.iloc[0],full.iloc[-1])
    assert np.isfinite(full.to_numpy()).all()

def test_feature_schema_rejects_missing_sensor():
    with pytest.raises(ValueError,match="Missing columns"):
        engineer_features(pd.DataFrame({"engine_id":[1],"cycle":[1]}),["sensor_1"])

def test_prefix_features_do_not_depend_on_future_failure_cycle():
    h=history().query("engine_id == 1")
    prefix=h[h.cycle <= 2]
    full=engineer_features(h,["sensor_1"])
    earlier=engineer_features(prefix,["sensor_1"])
    pd.testing.assert_frame_equal(full.loc[earlier.index],earlier)

def test_saved_preprocessor_matches_training_transform():
    h=history().query("engine_id == 1")
    pd.testing.assert_frame_equal(SequentialFeatureEngineer(["sensor_1"]).transform(h),engineer_features(h,["sensor_1"]))

def test_metrics_and_asymmetric_score():
    m=regression_metrics([1,2,3],[1,2,4])
    assert m["mae"]==pytest.approx(1/3)
    assert m["rmse"]==pytest.approx(1/np.sqrt(3))
    assert nasa_score([10],[8]) < nasa_score([10],[12])

def test_subset_selector_rejects_unknown_filesets():
    with pytest.raises(ValueError,match="FD001, FD002"):
        load_subset("FD005", "missing")
