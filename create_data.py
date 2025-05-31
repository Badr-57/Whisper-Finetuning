import argparse
import json
import unicodedata
from whisper.audio import N_FRAMES
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Union

import torch
import torchaudio
from tqdm import tqdm
from whisper.audio import load_audio
from whisper.audio import log_mel_spectrogram 
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import format_timestamp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a jsonl file to be used for fine-tuning a Whisper model"
    )
    parser.add_argument(
        "--with-timestamps",
        action="store_true",
        help=(
            "Read SRT (or VTT) files and audio files to create a jsonl file with timestamps and "
            "prompts for fine-tuning a Whisper model with time-aligned data. Defaults to True."
        ),
    )
    parser.add_argument(
        "--without-timestamps",
        action="store_false",
        dest="with_timestamps",
        help=(
            "Read a text file containing audio filenames and transcriptions to create a jsonl file "
            "without timestamps and prompts. This will be used for fine-tuning a Whisper model "
            "with utterance-by-utterance data"
        ),
    )
    parser.set_defaults(with_timestamps=True)

    parser.add_argument(
        "--audio-dir",
        type=str,
        help=(
            "Path to directory containing audio files. This option is used only when "
            "`--with-timestamps` is set. Audio formats that can be read by ffmpeg are supported."
        ),
    )
    parser.add_argument(
        "--transcript-dir",
        type=str,
        help=(
            "Path to directory containing transcripts in SRT (or VTT) format. This option is used "
            "only when `--with-timestamps` is set. Filenames must match the filenames under "
            "`--audio` directory except for the extension. For example, if the transcript file is "
            "`example.srt`, there must be an audio file like `example.wav` under `--audio` "
            "directory.",
        ),
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help=(
            "Path to a text file containing audio filenames and transcriptions. This option is "
            "used only when `--without-timestamps` is set. Each line must be in the format of "
            "`<audio_path>\t<transcription>`."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the data",
    )
    parser.add_argument("--output", type=str, default="data.json", help="Path to output json file")
    parser.add_argument(
        "--dump-dir", type=str, default="dump", help="Directory to dump audio features"
    )
    parser.add_argument(
        "--timestamp-resolution",
        type=int,
        default=20,
        help=(
            "Timestamp resolution in milliseconds. Defaults to 20ms. Since the native time "
            "resolution of Whisper tokens is 20ms, this option needs to be set to multiples of "
            "20ms."
        ),
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=223,
        help=(
            "Maximum length of prompt in Whisper tokens. Defaults to 223, which equals to "
            "`model.dims.n_text_ctx (=448) // 2 - 1` (-1 is for the special token `sot_prev` and "
            "the other half is for the transcribed tokens)."
        ),
    )
    parser.add_argument(
        "--max-tokens-length",
        type=int,
        default=219,
        help=(
            "Maximum length of text and timestamps tokens. Utterances longer than this will be "
            "skipped. Defaults to 219, which equals to `model.dims.n_text_ctx (=448) // 2 - 5` "
            "(5 is the maximum number of special tokens used other than the `sot_prev`."
        ),
    )
    parser.add_argument(
        "--subsampling-factor-for-silence",
        type=int,
        default=1,
        help=(
            "Subsampling factor for silence. This option is used to reduce the number of silence "
            "utterances. The original Whisper paper uses 1/10 of the number of silence utterances. "
            "Defaults to 1, which means no subsampling."
        ),
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="multilingual",
        choices=["multilingual", "english"],
        help=(
            "Type of Whisper tokenizer to use. Tokenizer is used to count the number of tokens "
            "in the transcriptions."
        ),
    )
    parser.add_argument("--normalize-unicode", action="store_true", help="Normalize unicode")
    return parser

DURATION = 30000  # 30 seconds in milliseconds
SAMPLE_RATE = 16000
DURATION_IN_SAMPLES = int(DURATION * SAMPLE_RATE / 1000)

@dataclass
class Utterance:
    """
    Representing a single segment of audio with a transcription. Corresponds to a single chunk in a
    .srt (or .vtt) file.
    """

    text: str
    start: Optional[int] = None  # in milliseconds
    end: Optional[int] = None  # in milliseconds

@dataclass
class Record:
    """
    A single training instance for Whisper.
    text can include timestamps in the format of <|0.00|>.
    """

    audio_path: str
    text: str  # text including timestamps
    language: str = "en"
    prompt: str = ""  # previous text including timestamps

@dataclass
class PromptNode:
    text: str  # text including timestamps
    num_tokens: int

class DataProcessor:
    def __init__(
        self,
        with_timestamps: bool = True,
        audio_dir: str = None,
        transcript_dir: str = None,
        data_file: str = None,
        language: str = "en",
        output: str = "data.json",
        dump_dir: str = "dump",
        timestamp_resolution: int = 20,
        max_prompt_length: int = 223,
        max_tokens_length: int = 219,
        subsampling_factor_for_silence: int = 1,
        tokenizer_type: str = "multilingual",
        normalize_unicode: bool = False,
    ) -> None:
        self.with_timestamps = with_timestamps
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        self.data_file = data_file
        self.language = language
        self.output = output
        self.dump_dir = dump_dir
        self.timestamp_resolution = timestamp_resolution
        self.max_prompt_length = max_prompt_length
        self.max_tokens_length = max_tokens_length
        self.subsampling_factor_for_silence = subsampling_factor_for_silence
        self.tokenizer_type = tokenizer_type
        self.normalize_unicode = normalize_unicode

        self._verify_args()

        self.tokenizer = get_tokenizer(multilingual=(self.tokenizer_type == "multilingual"))
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

    def _verify_args(self) -> None:
        if self.with_timestamps:
            if not self.audio_dir or not self.transcript_dir:
                raise ValueError(
                    "`audio_dir` and `transcript_dir` must be set when `with_timestamps` is True"
                )

            if self.timestamp_resolution % 20 != 0:
                raise ValueError(
                    "`timestamps_resolution` must be multiples of 20ms. "
                    f"Got {self.timestamp_resolution}"
                )
        else:
            if not self.data_file:
                raise ValueError("`data_file` must be set when `with_timestamps` is False")

        if self.language not in LANGUAGES:
            if self.language in TO_LANGUAGE_CODE:
                self.language = TO_LANGUAGE_CODE[self.language]
            else:
                raise ValueError(f"Unsupported language: {self.language}")

        if self.tokenizer_type not in ["multilingual", "english"]:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

        if Path(self.output).exists():
            raise ValueError(f"Output file {self.output} already exists")

    def run(self) -> None:
        if self.with_timestamps:
            self._process_with_timestamps()
        else:
            self._process_without_timestamps()

        if self.subsampling_factor_for_silence > 1:
            self._subsample_silence()

    def _process_without_timestamps(self) -> None:
        records = []
        with open(self.data_file, encoding="utf-8") as f:
            for line in f:
                audio_path, text = line.strip().split("\t")
                if self.normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)

                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.max_tokens_length:
                    print(
                        f"Skipping {audio_path} ({text}) because it is too long "
                        f"({len(tokens)} tokens)"
                    )
                    continue

                record = Record(audio_path=audio_path, text=text, language=self.language)
                records.append(record)

        self.write_records(records, self.output)

    def _process_with_timestamps(self) -> None:
        audio_paths = list(Path(self.audio_dir).iterdir())
        for audio_path in tqdm(audio_paths):
            speech_id = Path(audio_path).stem
            if (Path(self.transcript_dir) / f"{speech_id}.srt").exists():
                transcript_path = Path(self.transcript_dir) / f"{speech_id}.srt"
                try:
                    utterances_for_speech = self.read_utterances_from_srt(
                        transcript_path, self.normalize_unicode
                    )
                except Exception as e:
                    print(e)
                    print(f"Skipping {transcript_path} because of an error in the transcript")
                    continue
            elif (Path(self.transcript_dir) / f"{speech_id}.vtt").exists():
                transcript_path = Path(self.transcript_dir) / f"{speech_id}.vtt"
                try:
                    utterances_for_speech = self.read_utterances_from_vtt(
                        transcript_path, self.normalize_unicode
                    )
                except Exception as e:
                    print(e)
                    print(f"Skipping {transcript_path} because of an error in the transcript")
                    continue
            else:
                raise FileNotFoundError(f"Transcript file not found for {speech_id}")

            records = self._create_records_with_timestamps(utterances_for_speech, audio_path)
            self.write_records(records, self.output)

    @staticmethod
    def read_utterances_from_srt(
        transcript_path: Union[str, Path], normalize_unicode: bool = False
    ) -> List[Utterance]:
        utterances = []
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()
            timestamps_indices = [i for i, line in enumerate(lines) if " --> " in line]
            timestamps_indices.append(len(lines) + 1)  # dummy index to simplify loop

            for i in range(len(timestamps_indices) - 1):
                utterance_start = timestamps_indices[i]
                next_utterance_start = timestamps_indices[i + 1]

                start_time, end_time = lines[utterance_start].strip().split(" --> ")
                start_time = DataProcessor.str_to_milliseconds(start_time)
                end_time = DataProcessor.str_to_milliseconds(end_time)

                # Text is between [utterance_start + 1, next_utterance_start - 2)
                text = " ".join(
                    [line.strip() for line in lines[utterance_start + 1 : next_utterance_start - 2]]
                ).strip()
                if normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)
                if text == "":
                    continue

                utterances.append(Utterance(text=text, start=start_time, end=end_time))

        return utterances

    @staticmethod
    def read_utterances_from_vtt(
        transcript_path: Union[str, Path], normalize_unicode: bool = False
    ) -> List[Utterance]:
        utterances = []
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()
            timestamps_indices = [i for i, line in enumerate(lines) if " --> " in line]
            timestamps_indices.append(len(lines) + 1)  # dummy index to simplify loop

            for i in range(len(timestamps_indices) - 1):
                utterance_start = timestamps_indices[i]
                next_utterance_start = timestamps_indices[i + 1]

                start_time, end_time = lines[utterance_start].strip().split(" --> ")
                start_time = DataProcessor.str_to_milliseconds(start_time)
                end_time = DataProcessor.str_to_milliseconds(end_time)

                # Text is between [utterance_start + 1, next_utterance_start - 1)
                text = " ".join(
                    [line.strip() for line in lines[utterance_start + 1 : next_utterance_start - 1]]
                ).strip()
                if normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)
                if text == "":
                    continue

                utterances.append(Utterance(text=text, start=start_time, end=end_time))

        return utterances
    
    def _create_records_with_timestamps(
        self, utterances: List[Utterance], audio_path: Path
    ) -> List[Record]:
        with torch.inference_mode():
            audio = torch.tensor(load_audio(audio_path))
        dump_dir = Path(self.dump_dir) / audio_path.stem
        dump_dir.mkdir(parents=True, exist_ok=True)
        records = []
        prompt_buffer: Deque[PromptNode] = deque()
        segment_start, segment_end = 0, DURATION  # in milliseconds
    
        idx = 0
        segment_count = 0
        max_segments = 5000  # Safety limit to prevent infinite loops
        
        while idx < len(utterances) and segment_count < max_segments:
            segment_count += 1  # Safety counter
            
            # Skip utterances longer than segment
            if utterances[idx].end - utterances[idx].start > DURATION:
                print(f"Skipping long utterance: {utterances[idx].text} "
                      f"({utterances[idx].start}-{utterances[idx].end}ms)")
                idx += 1
                segment_start = utterances[idx].end if idx < len(utterances) else segment_start + DURATION
                segment_end = segment_start + DURATION
                continue
    
            segment_audio_path = self._save_segment_features(audio, segment_start, dump_dir)
            
            # Skip segment if audio processing failed
            if segment_audio_path is None:
                segment_start += DURATION
                segment_end = segment_start + DURATION
                continue
    
            prompt = self._get_prompt(prompt_buffer)
    
            segment_utterances = []
            while idx < len(utterances) and utterances[idx].start < segment_end:
                segment_utterances.append(utterances[idx])
                idx += 1
    
            # Skip segment if it contains invalid utterances
            if not self._is_valid_utterances(segment_utterances, segment_start):
                tqdm.write(f"Skipping {audio_path} ({segment_start}-{segment_end}ms) - invalid utterances")
                prompt_buffer.clear()
                if segment_utterances:
                    segment_start = segment_utterances[-1].end
                else:
                    segment_start += DURATION
                segment_end = segment_start + DURATION
                continue
    
            tokens_length = 0
            segment_text = []
            for utterance in segment_utterances:
                try:
                    start_token = self._get_time_token(utterance.start, segment_start, audio_path)
                except ValueError as e:
                    tqdm.write(f"Skipping utterance: {str(e)}")
                    continue
                    
                if utterance.end <= segment_end:
                    try:
                        end_token = self._get_time_token(utterance.end, segment_start, audio_path)
                    except ValueError:
                        # Handle case where end is slightly beyond segment
                        end_token = ""
                    
                    utterance_text = self._add_leading_space(utterance.text)
                    segment_text.extend([start_token, utterance_text, end_token])
                    new_prompt_length = len(self.tokenizer.encode(utterance_text)) + 2
                    new_prompt_node = PromptNode(
                        start_token + utterance_text + end_token, new_prompt_length
                    )
                    tokens_length += new_prompt_length
                else:
                    segment_text.append(start_token)
                    new_prompt_node = PromptNode(start_token, 1)
                    tokens_length += 1
    
                prompt_buffer.append(new_prompt_node)
    
            if tokens_length > self.max_tokens_length:
                tqdm.write(f"Skipping {audio_path} ({segment_start}-{segment_end}ms) - too long ({tokens_length} tokens)")
            else:
                record = Record(
                    audio_path=segment_audio_path,
                    language=self.language,
                    text="".join(segment_text),
                    prompt=prompt,
                )
                records.append(record)
    
            # Always advance the segment to prevent infinite loops
            if len(segment_utterances) == 0:
                segment_start += DURATION
            elif segment_utterances[-1].end <= segment_end:
                segment_start = segment_utterances[-1].end
            else:
                # Only reprocess if we have valid utterances
                if segment_utterances:
                    segment_start = segment_utterances[-1].start
                    if idx > 0:
                        idx -= 1  # Reprocess current utterance in next segment
                else:
                    segment_start += DURATION
                    
            segment_end = segment_start + DURATION
    
        if segment_count >= max_segments:
            tqdm.write(f"Warning: Exceeded maximum segments for {audio_path}. Processed {segment_count} segments.")
        
        return records
    
    def _save_segment_features(self, audio: torch.Tensor, segment_start: int, dump_dir: Path) -> Union[str, None]:
        from whisper import log_mel_spectrogram
        import numpy as np
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        audio_start_idx = int(segment_start * SAMPLE_RATE / 1000)
        segment_end_idx = min(audio_start_idx + DURATION_IN_SAMPLES, audio.size(0))
        
        if segment_end_idx <= audio_start_idx:
            return None
        
        segment_audio = audio[audio_start_idx:segment_end_idx].to(device)
        
        if len(segment_audio) < 400:
            return None
            
        try:
            segment_audio_np = segment_audio.cpu().numpy().astype(np.float32)
            mel = log_mel_spectrogram(segment_audio_np, n_mels=128, padding=N_FRAMES if segment_audio_np.size > 0 else 0)
            
            # Dynamic padding
            if mel.shape[1] < N_FRAMES:
                padded = torch.zeros((mel.shape[0], N_FRAMES), 
                                    dtype=mel.dtype, 
                                    device=device)
                padded[:, :mel.shape[1]] = mel
                mel = padded
            elif mel.shape[1] > N_FRAMES:
                mel = mel[:, :N_FRAMES]
                
            # Save directly from GPU
            segment_features_path = dump_dir / f"{segment_start}.pt"
            torch.save(mel.cpu(), segment_features_path)  # Save to CPU for later loading
            return str(segment_features_path.absolute())
            
        except Exception as e:
            tqdm.write(f"Feature extraction failed: {str(e)}")
            return None
    
    def _is_valid_utterances(self, utterances: List[Utterance], segment_start: int) -> bool:
        if len(utterances) == 0:
            return True

        for utterance in utterances:
            # Check utterances are within segment and non-overlapping
            if utterance.start < segment_start:
                return False
            if utterance.start > utterance.end:
                return False

        for i in range(len(utterances) - 1):
            if utterances[i].end > utterances[i + 1].start:
                return False

        return True

    def _add_leading_space(self, text: str) -> str:
        """
        Add leading space for languages that use word separators
        """
        if self.language in ["zh", "ja", "th", "lo", "my"]:
            return text
        return " " + text

    @staticmethod
    def str_to_milliseconds(s: str) -> int:
        """
        Convert timestamp string to milliseconds
        """
        if "," in s:
            time, milliseconds = s.split(",")
        elif "." in s:
            time, milliseconds = s.split(".")
        else:
            raise ValueError(f"Invalid time format: {s}")

        hours, minutes, seconds = time.split(":")
        return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)

    def _get_time_token(self, time: int, segment_start: int, audio_path: Path) -> str:
        """
        Get time token for given time
        """
        if time < segment_start or segment_start + DURATION < time:
            raise ValueError(
                f"Time {format_timestamp(time / 1000)} is out of segment "
                f"({format_timestamp(segment_start / 1000)} - "
                f"{format_timestamp((segment_start + DURATION) / 1000)}) of {audio_path}"
            )

        time_in_segment = max(0, min(DURATION, time - segment_start))
        nearest_timestamp = round(time_in_segment / self.timestamp_resolution) * self.timestamp_resolution
        return f"<|{nearest_timestamp / 1000:.2f}|>"

    def _get_prompt(self, prompt_buffer: Deque[PromptNode]) -> str:
        prompt_length = 0
        prompt_buffer_idx = len(prompt_buffer)
        while prompt_buffer_idx >= 1 and prompt_length < self.max_prompt_length:
            prompt_buffer_idx -= 1
            prompt_length += prompt_buffer[prompt_buffer_idx].num_tokens

        for _ in range(prompt_buffer_idx):
            prompt_buffer.popleft()

        return "".join([node.text for node in prompt_buffer])

    @staticmethod
    def read_records(path: Union[str, Path]) -> List[Record]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                record = Record(
                    audio_path=data["audio_path"],
                    text=data["text"],
                    language=data["language"],
                    prompt=data["prompt"],
                )
                records.append(record)
        return records

    @staticmethod
    def write_records(records: List[Record], path: Union[str, Path]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                data = {
                    "audio_path": record.audio_path,
                    "text": record.text,
                    "language": record.language,
                    "prompt": record.prompt,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _subsample_silence(self) -> None:
        records = self.read_records(self.output)

        silence_records = [r for r in records if r.text == ""]
        non_silence_records = [r for r in records if r.text != ""]
        filtered_records = non_silence_records + silence_records[:: self.subsampling_factor_for_silence]

        Path(self.output).unlink()
        self.write_records(filtered_records, self.output)

def main():
    args = get_parser().parse_args()
    processor = DataProcessor(
        with_timestamps=args.with_timestamps,
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir,
        data_file=args.data_file,
        language=args.language,
        output=args.output,
        dump_dir=args.dump_dir,
        timestamp_resolution=args.timestamp_resolution,
        max_prompt_length=args.max_prompt_length,
        max_tokens_length=args.max_tokens_length,
        subsampling_factor_for_silence=args.subsampling_factor_for_silence,
        tokenizer_type=args.tokenizer_type,
        normalize_unicode=args.normalize_unicode,
    )
    processor.run()

if __name__ == "__main__":
    main()