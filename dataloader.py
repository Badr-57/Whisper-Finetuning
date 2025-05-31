import re
from typing import List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from whisper.audio import CHUNK_LENGTH, N_FRAMES, pad_or_trim
from whisper.audio import log_mel_spectrogram
from whisper.tokenizer import Tokenizer
from tqdm import tqdm

from create_data import DataProcessor, Record

def collate_fn(data):
    x, y_in, y_out = zip(*data)
    
    # Dynamic padding for variable-length audio
    max_len = max([x_i.shape[1] for x_i in x])
    x_padded = []
    for x_i in x:
        pad_amount = max_len - x_i.shape[1]
        x_padded.append(torch.nn.functional.pad(x_i, (0, pad_amount)))
    x = torch.stack(x_padded)
    
    y_in = pad_sequence(y_in, batch_first=True, padding_value=0)
    y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)
    
    return x, y_in, y_out

class CachedDataset(Dataset):
    """Simple in-memory cache wrapper for any dataset"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = [None] * len(dataset)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        if self.cache[index] is None:
            self.cache[index] = self.dataset[index]
        return self.cache[index]

class AudioDataset(Dataset):
    def __init__(
        self,
        records: List[Record],
        tokenizer: Tokenizer,
        fp16: bool = True,
        no_timestamps_training: bool = False,
        max_prompt_length: int = 223,
        prompt_use_rate: float = 0.5,
        no_timestamps_rate: float = 0.5,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.fp16 = fp16
        self.no_timestamps_training = no_timestamps_training
        self.max_prompt_length = max_prompt_length
        self.prompt_use_rate = prompt_use_rate
        self.no_timestamps_rate = no_timestamps_rate

        # Get context size dynamically from tokenizer
        self.model_n_text_ctx = tokenizer.model.n_text_ctx if hasattr(tokenizer, 'model') else 448
        self.num_frames_per_second = N_FRAMES / CHUNK_LENGTH
        
        # Enhanced timestamp pattern
        self.timestamp_pattern = re.compile(r"(<\|[0-9]{1,2}\.[0-9]{2}\|>)")

    def __len__(self) -> int:
        return len(self.records)

    def _get_prompt_tokens(self, prompt: str) -> List[int]:
        if len(prompt) > 0 and torch.rand(1) < self.prompt_use_rate:
            prompt_tokens = self._encode_text_with_timestamps(prompt)[-self.max_prompt_length :]
            prompt_tokens = [self.tokenizer.sot_prev] + prompt_tokens
        else:
            prompt_tokens = []
        return prompt_tokens

    def _get_special_tokens(
        self, is_text_empty: bool, language: str, no_timestamps: bool
    ) -> List[int]:
        if is_text_empty:
            return [self.tokenizer.sot, self.tokenizer.no_speech]
        else:
            tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.special_tokens["<|transcribe|>"],
            ]
            if no_timestamps:
                tokens.append(self.tokenizer.no_timestamps)
            return tokens

    def _encode_text_with_timestamps(self, text: str) -> List[int]:
        parts = self.timestamp_pattern.split(text)
        parts = [token for token in parts if token != ""]
        tokens = []
        for part in parts:
            if self.timestamp_pattern.fullmatch(part) is not None:
                timestamp = float(part[2:-2])
                token = self.tokenizer.timestamp_begin + round(timestamp * 100) // 2
                tokens.append(token)
            else:
                tokens.extend(self.tokenizer.encode(part))
        return tokens

    def _get_partial_segment_start(self, tokens: List[int]) -> Optional[float]:
        if (
            len(tokens) >= 2
            and tokens[-2] >= self.tokenizer.timestamp_begin
            and tokens[-1] >= self.tokenizer.timestamp_begin
        ):
            return (tokens[-1] - self.tokenizer.timestamp_begin) * 0.02
        return None

    def _get_text_tokens(self, text: str, no_timestamps: bool) -> Tuple[List[int], Optional[float]]:
        text_tokens = self._encode_text_with_timestamps(text)
        next_partial_segment_start = self._get_partial_segment_start(text_tokens)
        if no_timestamps:
            text_tokens = [t for t in text_tokens if t < self.tokenizer.timestamp_begin]
        return text_tokens, next_partial_segment_start

    def _calculate_mel(
        self, audio_path: str, next_partial_segment_start: Optional[float], no_timestamps: bool
    ) -> torch.Tensor:
        # Handle precomputed features
        if audio_path.endswith('.pt'):
            mel = torch.load(audio_path)
        else:
            mel = log_mel_spectrogram(audio_path, n_mels=128)
            
        if no_timestamps and next_partial_segment_start is not None:
            mel = mel[:, : int(next_partial_segment_start * self.num_frames_per_second)]
            
        # Dynamic padding to max length
        max_len = max(mel.shape[1], N_FRAMES)
        mel = pad_or_trim(mel, max_len)
        
        if self.fp16:
            mel = mel.half()
        return mel

    def _construct_decoder_output(
        self, prompt_tokens: List[int], special_tokens: List[int], text_tokens: List[int]
    ) -> List[int]:
        if len(prompt_tokens) == 0:
            return special_tokens[1:] + text_tokens + [self.tokenizer.eot]
        else:
            return (
                [-100] * (len(prompt_tokens) - 1)
                + special_tokens
                + text_tokens
                + [self.tokenizer.eot]
            )
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        no_timestamps = self.no_timestamps_training or torch.rand(1) < self.no_timestamps_rate

        prompt_tokens = self._get_prompt_tokens(record.prompt)
        text_tokens, next_partial_segment_start = self._get_text_tokens(record.text, no_timestamps)
        
        max_allowed = self.model_n_text_ctx - 5  # Reserve space for special tokens
        total_tokens = len(prompt_tokens) + len(text_tokens)
        
        if total_tokens > max_allowed:
            # Prioritize keeping current text over prompt
            keep_text = min(len(text_tokens), max_allowed)
            keep_prompt = max_allowed - keep_text
            
            text_tokens = text_tokens[:keep_text]
            prompt_tokens = prompt_tokens[-keep_prompt:] if keep_prompt > 0 else []
            
            tqdm.write(f"Truncated record {record.audio_path} from {total_tokens} to {max_allowed} tokens")
    
        # Rebuild special tokens after potential truncation
        is_text_empty = len(text_tokens) == 0
        special_tokens = self._get_special_tokens(is_text_empty, record.language, no_timestamps)

        decoder_input = prompt_tokens + special_tokens + text_tokens
        if len(decoder_input) > self.model_n_text_ctx:
            raise ValueError(f"Input too long: {record} ({len(decoder_input)} tokens)")

        decoder_output = self._construct_decoder_output(prompt_tokens, special_tokens, text_tokens)
        mel = self._calculate_mel(record.audio_path, next_partial_segment_start, no_timestamps)

        return (
            mel,
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(decoder_output, dtype=torch.long),
        )

# Changed function name from get_dataloader to get_dataset
def get_dataset(
    json: str,
    tokenizer: Tokenizer,
    fp16: bool = True,
    no_timestamps_training: bool = False,
    max_prompt_length: int = 223,
    prompt_use_rate: float = 0.5,
    no_timestamps_rate: float = 0.5,
) -> Dataset:
    records = DataProcessor.read_records(json)

    # Create base dataset
    base_dataset = AudioDataset(
        records,
        tokenizer,
        fp16=fp16,
        no_timestamps_training=no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=prompt_use_rate,
        no_timestamps_rate=no_timestamps_rate,
    )
    
    # Wrap with caching
    return CachedDataset(base_dataset)