"""Raw counter exports must retain units and reject ambiguous launches."""
from pathlib import Path

import pytest

from benchmarks.cuda_gemm_counter_study import read_ncu_csv


def test_raw_csv_preserves_units_and_quoted_kernel(tmp_path: Path) -> None:
    path=tmp_path/"counter.csv"
    path.write_text('"Kernel Name","gpu__time_duration.sum"\n"","ms"\n"kernel<a,b>","3.2"\n',encoding="utf-8")
    result=read_ncu_csv(path)
    assert result["kernel"]=="kernel<a,b>"
    assert result["units"]["gpu__time_duration.sum"]=="ms"
    assert result["metrics"]["gpu__time_duration.sum"]=="3.2"


def test_multiple_kernels_are_not_silently_aggregated(tmp_path: Path) -> None:
    path=tmp_path/"counter.csv"
    path.write_text('"Kernel Name","value"\n"","%"\n"one","1"\n"two","2"\n',encoding="utf-8")
    with pytest.raises(ValueError,match="one kernel"):
        read_ncu_csv(path)
