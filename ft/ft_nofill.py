import atexit
import sys

import freqtrade.data.dataprovider as DP
import freqtrade.data.history as H

CALLS = dict(load_data=0, dataprovider_load_pair_history=0)
_orig_load_data = H.load_data
_orig_dp_lph = DP.load_pair_history


def load_data_nofill(*args, **kwargs):
    kwargs["fill_up_missing"] = False
    CALLS["load_data"] += 1
    return _orig_load_data(*args, **kwargs)


def dp_load_pair_history_nofill(*args, **kwargs):
    kwargs["fill_up_missing"] = False
    CALLS["dataprovider_load_pair_history"] += 1
    return _orig_dp_lph(*args, **kwargs)


H.load_data = load_data_nofill
DP.load_pair_history = dp_load_pair_history_nofill


@atexit.register
def _report():
    print(f"ft_nofill: fill_up_missing forced False; calls {CALLS}", file=sys.stderr)


from freqtrade.main import main

main(sys.argv[1:])
