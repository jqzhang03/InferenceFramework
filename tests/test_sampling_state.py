import pytest
import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sampling_state import SamplingState
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from test_scheduler import _make_scheduler


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_sampling_rows_follow_order_and_verify_width() -> None:
    state = SamplingState(torch.device("cuda"))
    a, b = [Sequence([1], SamplingParams(temperature=t)) for t in (0.25, 0.75)]
    batch = state.prepare([a, b], varying_rows=False)
    assert batch.tolist() == [0.25, 0.75]
    assert state.prepare([a, b], varying_rows=False) is batch

    b.draft_tokens = [2, 3]
    b.num_scheduled_tokens = 3
    assert state.prepare([b, a], varying_rows=True).tolist() == [0.75, 0.75, 0.75, 0.25]
    b.draft_tokens = []
    b.num_scheduled_tokens = 1
    assert state.prepare([b, a], varying_rows=True).tolist() == [0.75, 0.25]


def test_sampling_state_survives_temporarily_unscheduled_request() -> None:
    state = SamplingState(torch.device("cuda"))
    a, b, c = [Sequence([1], SamplingParams(temperature=t)) for t in (0.25, 0.5, 0.75)]
    state.prepare([a, b], varying_rows=False)
    assert state.prepare([c, b], varying_rows=False).tolist() == [0.75, 0.5]
    assert a.seq_id in state._temperatures
    state.release(c.seq_id)
    assert state._batch_temperatures is None
    assert state.prepare([b, a], varying_rows=False).tolist() == [0.5, 0.25]


@pytest.mark.parametrize("kv_swap", [False, True])
def test_preemption_releases_sampling_state_until_resume(kv_swap: bool) -> None:
    state = SamplingState(torch.device("cuda"))
    scheduler = _make_scheduler(kv_swap=kv_swap, recompute_aware_preemption=False)
    scheduler.on_request_removed = state.release
    seq = Sequence([1] * 256, SamplingParams(temperature=0.5, max_tokens=2, ignore_eos=True))
    scheduler.add(seq)
    scheduled, _ = scheduler.schedule()
    state.prepare(scheduled, varying_rows=False)
    scheduler.postprocess(scheduled, [10])
    seq.is_prefill = False
    scheduler.running.remove(seq)

    scheduler.preempt(seq)
    assert not state._temperatures
    assert state._batch_temperatures is None
    assert seq.swapped == kv_swap
    if kv_swap:
        scheduler.finish_swap_out(seq, scheduler.swap_pairs[-1][1])
    scheduled, _ = scheduler.schedule()
    if kv_swap:
        scheduler.finish_swap_in(seq)
    assert scheduled == [seq]
    assert not state._temperatures
    assert state.prepare(scheduled, varying_rows=False).tolist() == [0.5]
    scheduler.postprocess(scheduled, [11])
    assert seq.is_finished
    assert not state._temperatures


def test_cancellation_releases_sampling_state() -> None:
    engine = LLMEngine.__new__(LLMEngine)
    engine._pd = False
    engine.scheduler = _make_scheduler()
    state = SamplingState(torch.device("cuda"))
    engine.scheduler.on_request_removed = state.release
    seq = Sequence([1], SamplingParams(temperature=0.5))
    engine.scheduler.add(seq)
    scheduled, _ = engine.scheduler.schedule()
    state.prepare(scheduled, varying_rows=False)

    assert engine.cancel_request(seq.seq_id)
    assert seq.is_finished
    assert not state._temperatures
    assert state._batch_temperatures is None
