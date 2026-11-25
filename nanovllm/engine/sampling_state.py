import torch

from nanovllm.engine.sequence import Sequence


class SamplingState:
    """Per-request sampling metadata and the last batch's ordered device input."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._temperatures: dict[int, float] = {}
        self._batch: list[Sequence] = []
        self._row_counts: list[int] | None = None
        self._batch_temperatures: torch.Tensor | None = None

    def release(self, seq_id: int) -> None:
        if self._temperatures.pop(seq_id, None) is not None:
            self._batch.clear()
            self._row_counts = None
            self._batch_temperatures = None

    def prepare(self, seqs: list[Sequence], *, varying_rows: bool) -> torch.Tensor:
        row_counts = None
        if varying_rows:
            row_counts = [1 if s.draft_tokens is None else s.num_scheduled_tokens for s in seqs]
        if (self._batch_temperatures is not None and self._batch == seqs
                and self._row_counts == row_counts):
            return self._batch_temperatures

        # These are small scalars. Assemble the ordered input on the host to
        # avoid GPU gather launches when requests join, leave or change order.
        temperatures = []
        for seq in seqs:
            temperature = self._temperatures.get(seq.seq_id)
            if temperature is None:
                temperature = seq.temperature
                self._temperatures[seq.seq_id] = temperature
            temperatures.append(temperature)
        if row_counts is not None:
            temperatures = [t for t, count in zip(temperatures, row_counts) for _ in range(count)]
        self._batch_temperatures = torch.tensor(
            temperatures, dtype=torch.float32, device="cpu", pin_memory=True,
        ).to(self._device, non_blocking=True)
        self._batch = seqs.copy()
        self._row_counts = row_counts
        return self._batch_temperatures
