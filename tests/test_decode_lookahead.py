from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context, set_context


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_lookahead_packs_the_next_position_and_reuses_page_table() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 256
    first, second = Sequence([1] * 255), Sequence([2] * 260)
    first.block_table = [4]
    second.block_table = [5, 6]
    tokens = torch.tensor([17, 19], device="cuda")
    tables = torch.tensor([[4, -1], [5, 6]], dtype=torch.int32, device="cuda")
    set_context(False, block_tables=tables)
    try:
        staged = runner.prepare_decode_lookahead([first, second], tokens)
        assert staged is not None
        assert staged.positions.tolist() == [255, 260]
        assert staged.context_lens.tolist() == [256, 261]
        assert staged.slot_mapping.tolist() == [4 * 256 + 255, 6 * 256 + 4]
        assert staged.block_tables is tables and staged.input_ids is tokens

        first.append_token(17)
        assert runner.prepare_decode_lookahead([first, second], tokens) is None
        second.max_tokens = 1
        assert runner.prepare_decode_lookahead([second], tokens[1:]) is None
    finally:
        reset_context()


@pytest.fixture(scope="module")
def engine() -> Iterator[LLM]:
    model = Path.home() / "huggingface/Qwen3-0.6B"
    if not model.is_dir():
        pytest.skip("local Qwen3-0.6B checkpoint required")
    llm = LLM(str(model), max_num_seqs=4, max_num_batched_tokens=512,
              max_model_len=512, gpu_memory_utilization=0.55, execution_mode="mixed",
              latency_aware_scheduling=False, slo_aware_scheduling=False,
              tpot_aware_scheduling=False)
    llm.scheduler.block_manager.no_share = True
    try:
        yield llm
    finally:
        llm.exit()


def test_decode_lookahead_graph_parity_across_block_boundary(
    engine: LLM, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = engine.model_runner
    original = runner.run
    original_prepare = runner.prepare_decode_lookahead
    staged_positions = []
    rejected_boundary = False

    def run(seqs, kind, return_logits=False, return_hidden=False,
            decode_input=None, prepare_next_decode=False):
        if decode_input is not None:
            staged_positions.append([len(seq) - 1 for seq in seqs])
        return original(seqs, kind, return_logits, return_hidden,
                        decode_input, prepare_next_decode)

    def prepare(seqs, tokens):
        nonlocal rejected_boundary
        staged = original_prepare(seqs, tokens)
        if any(len(seq) == runner.block_size for seq in seqs):
            assert staged is None
            rejected_boundary = True
        return staged

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner, "prepare_decode_lookahead", prepare)
    prompts = [list(range(102, 355)), list(range(500, 753))]
    params = [SamplingParams(temperature=t, max_tokens=12, ignore_eos=True) for t in (0.6, 0.9)]
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(runner, "decode_lookahead_enabled", enabled)
        torch.manual_seed(17)
        out = engine.generate(prompts, params, use_tqdm=False)
        results.append([row["token_ids"] for row in out])
    assert results[0] == results[1]
    assert rejected_boundary
    assert staged_positions and all(256 not in row for row in staged_positions)
    assert engine.model_runner.decode_cudagraph_replay_count > 0


@pytest.mark.parametrize("arrival", [False, True])
def test_decode_lookahead_is_discarded_on_eos_and_arrival(
    engine: LLM, monkeypatch: pytest.MonkeyPatch, arrival: bool,
) -> None:
    runner = engine.model_runner
    original = runner.run
    original_prepare = runner.prepare_decode_lookahead
    phase = "prefill"
    saw_decode = False
    incoming = arrival
    runs = []

    def run(seqs, kind, return_logits=False, return_hidden=False,
            decode_input=None, prepare_next_decode=False):
        nonlocal phase
        phase = kind
        runs.append((kind, len(seqs), decode_input is not None))
        return original(seqs, kind, return_logits, return_hidden,
                        decode_input, prepare_next_decode)

    def sample(logits, temperatures):
        nonlocal saw_decode
        tokens = [12] * len(logits)
        if phase == "decode" and not saw_decode:
            saw_decode = True
            tokens[0] = engine.config.eos
        return torch.tensor(tokens, dtype=torch.int64, device=logits.device)

    def prepare(seqs, tokens):
        nonlocal incoming
        staged = original_prepare(seqs, tokens)
        if incoming and staged is not None:
            engine.request_decode_burst_yield()
            incoming = False
        return staged

    monkeypatch.setattr(runner, "run", run)
    monkeypatch.setattr(runner.sampler, "forward", sample)
    monkeypatch.setattr(runner, "prepare_decode_lookahead", prepare)
    params = SamplingParams(temperature=0.8, max_tokens=7)
    first = engine.add_request([100, 101], params)
    second = engine.add_request([102, 103], params)
    try:
        _, kind, *_ = engine.step()
        assert kind == "prefill"
        finished, kind, *_ = engine.step()
        assert kind == "decode" and [seq_id for seq_id, _ in finished] == [first]
        assert engine._last_step_decode_burst_yielded == arrival
        if arrival:
            assert runs == [("prefill", 2, False), ("decode", 2, False)]
            engine.clear_decode_burst_yield()
        while not engine.is_finished():
            finished, *_ = engine.step()
        assert runs[2] == ("decode", 1, False)
        assert finished == [(second, [12] * 7)]
    finally:
        engine.clear_decode_burst_yield()
        engine.cancel_request(first)
        engine.cancel_request(second)
