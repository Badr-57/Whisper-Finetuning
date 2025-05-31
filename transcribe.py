import argparse
from pathlib import Path
from typing import Union

import torch
import whisper
from tqdm import tqdm
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import get_writer

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--save-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--language", type=str, default="en", choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="Audio language")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--model", default="large", help="Whisper model name or path")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="Transcription or translation")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for transcription")
    return parser

def main():
    args = get_parser().parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model with dynamic configuration
    model = whisper.load_model(args.model, args.device)
    writer = get_writer("srt", args.save_dir)
    
    audio_files = list(Path(args.audio_dir).iterdir())
    batches = [audio_files[i:i + args.batch_size] for i in range(0, len(audio_files), args.batch_size)]
    
    for batch in tqdm(batches):
        results = model.transcribe(
            [str(path) for path in batch],
            task=args.task,
            language=args.language,
            batch_size=args.batch_size
        )
        
        for audio_path, result in zip(batch, results):
            writer(result, str(audio_path))

if __name__ == "__main__":
    main()