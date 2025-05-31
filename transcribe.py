import argparse
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import whisper
from tqdm import tqdm
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from whisper.utils import get_writer

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("--audio-dir", type=str, required=True, 
                        help="Directory containing audio files")
    parser.add_argument("--language", type=str, default="ar",
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="Language of the audio content")
    parser.add_argument("--model", default="large-v3",
                        help="Name/path of Whisper model or HuggingFace repo")
    parser.add_argument("--task", type=str, default="transcribe",
                        choices=["transcribe", "translate"],
                        help="Task to perform")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of files to process simultaneously")
    parser.add_argument("--chunk-size", type=int, default=30,
                        help="Audio chunk size in seconds for processing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for inference")
    parser.add_argument("--precision", default="fp16",
                        choices=["fp16", "fp32", "int8"],
                        help="Numerical precision for inference")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for deterministic)")
    parser.add_argument("--suppress-silence", action="store_true",
                        help="Enable silence suppression using VAD")
    parser.add_argument("--word-timestamps", action="store_true",
                        help="Generate word-level timestamps")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size for beam search decoding")
    return parser

def transcribe_audio_batch(
    model: whisper.Whisper,
    audio_paths: List[Path],
    task: str,
    language: str,
    temperature: float,
    suppress_silence: bool,
    word_timestamps: bool,
    beam_size: int
) -> list:
    """Transcribe a batch of audio files with optimized settings"""
    return model.transcribe(
        [str(p) for p in audio_paths],
        task=task,
        language=language,
        temperature=temperature,
        suppress_silence=suppress_silence,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
        batch_size=len(audio_paths)
    )

def main():
    args = get_parser().parse_args()
    
    # Create output directory
    output_dir = Path(args.audio_dir) / "SRTs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = whisper.load_model(args.model, device=args.device)
    
    # Enable multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for transcription")
    
    # Precision configuration
    if args.precision == "fp16" and "cuda" in args.device:
        model = model.half()
    elif args.precision == "int8":
        model = model.to(torch.int8)
    
    # Collect audio files
    audio_exts = [".mp3", ".wav", ".flac", ".m4a"]
    audio_files = [f for f in Path(args.audio_dir).iterdir() 
                  if f.is_file() and f.suffix.lower() in audio_exts]
    
    # Filter out processed files
    processed_count = 0
    unprocessed_files = []
    for audio_file in audio_files:
        srt_path = output_dir / f"{audio_file.stem}.srt"
        if srt_path.exists():
            processed_count += 1
        else:
            unprocessed_files.append(audio_file)
    
    print(f"Found {len(audio_files)} audio files ({processed_count} already processed)")
    
    # Batch processing
    writer = get_writer("srt", str(output_dir))
    batch_size = args.batch_size
    max_attempts = 5
    
    progress_bar = tqdm(total=len(unprocessed_files), desc="Transcribing")
    
    while unprocessed_files:
        batch_files = unprocessed_files[:batch_size]
        attempt = 0
        success = False
        
        while attempt < max_attempts and not success:
            try:
                results = transcribe_audio_batch(
                    model=model,
                    audio_paths=batch_files,
                    task=args.task,
                    language=args.language,
                    temperature=args.temperature,
                    suppress_silence=args.suppress_silence,
                    word_timestamps=args.word_timestamps,
                    beam_size=args.beam_size
                )
                
                # Save results
                for audio_path, result in zip(batch_files, results):
                    writer(result, str(audio_path))
                
                # Update progress
                progress_bar.update(len(batch_files))
                unprocessed_files = unprocessed_files[len(batch_files):]
                success = True
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    attempt += 1
                    new_size = max(batch_size // 2, 1)
                    print(f"OOM with batch size {batch_size}, reducing to {new_size}")
                    batch_size = new_size
                else:
                    raise
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Skip problematic files
                unprocessed_files = unprocessed_files[len(batch_files):]
                break
    
    progress_bar.close()
    print(f"Transcription complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()