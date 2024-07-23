import tempfile
import sounddevice as sd
import subprocess
import wave
import os
import queue
import threading
import re
from datetime import timedelta
import requests

def run_transcription(command: list, verbose: int) -> str:
    """
    Executes the transcription command and returns the output.

    Args:
        command (list): The command to run the transcription.
        verbose (int): Verbosity level. If 1, prints detailed output; if 0, prints minimal output.

    Returns:
        str: The transcription output.
    """
    if verbose:
        print(f"Executing command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)

    if verbose:
        print(f"Return code: {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise RuntimeError(f"Transcription failed with return code {result.returncode}")

def transcribe_to_txt(input_filename: str, model_string='ggml-small.en-tdrz.bin', verbose=1, use_tinydiarize=True) -> str:
    """
    Transcribes an audio file to text using the whisper transcription model.

    Args:
        input_filename (str): Path to the input audio file.
        model_string (str): Name of the model to use for transcription.
        verbose (int): Verbosity level. If 1, prints detailed output; if 0, prints minimal output.
        use_tinydiarize (bool): If True, enables tinydiarize for speaker diarization.

    Returns:
        str: The transcription output.
    """
    print('Running whisper transcription...')

    main_component_path = '/Users/carlos.salas/Documents/sl-vista-backend/whisper_cpp/main'
    model_path = f'/Users/carlos.salas/Documents/sl-vista-backend/whisper_cpp/models/{model_string}'
    command = [main_component_path, '-m', model_path, '-f', input_filename]

    # Add tinydiarize flag if enabled
    if use_tinydiarize:
        command.append('-tdrz')

    try:
        transcription = run_transcription(command, verbose)
        print("Transcription successful. Output:")
        if transcription:
            print(transcription)
            # payload = {"input_chunk": transcription}
            # classification_reply = requests.post("http://0.0.0.0/8080/classify-chunk", data=payload)
            # prediction = classification_reply.json()
            # if prediction == 'pricing':
        else:
            print("No transcription output (possibly due to short audio)")
        return transcription
    except RuntimeError as e:
        print(f"Error during transcription: {e}")
        return ""

def process_audio_chunk(audio_queue, samplerate: int, model_string: str, verbose: int, use_tinydiarize: bool, output_file: str, start_time: float) -> None:
    cumulative_duration = start_time
    
    while True:
        indata = audio_queue.get()
        if indata is None:  # Signal to stop processing
            break

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix='audio_', dir='.') as tmpfile:
            with wave.open(tmpfile.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono audio
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(samplerate)
                wav_file.writeframes(indata)

            try:
                transcription = transcribe_to_txt(tmpfile.name, model_string=model_string, verbose=verbose, use_tinydiarize=use_tinydiarize)
                
                # Correct timing and save to file
                chunk_duration = len(indata) / samplerate
                corrected_transcription = correct_timing(transcription, cumulative_duration, chunk_duration)
                
                with open(output_file, 'a') as f:
                    f.write(corrected_transcription + '\n')
                
                cumulative_duration += chunk_duration
            except Exception as e:
                print(f"Error during transcription: {e}")

            os.remove(tmpfile.name)

def correct_timing(transcription: str, start_time: float, duration: float) -> str:
    """Corrects the timing in the transcription."""
    def format_time(seconds):
        return str(timedelta(seconds=seconds)).rstrip('0').rstrip('.')

    end_time = start_time + duration
    time_str = f"[{format_time(start_time)} --> {format_time(end_time)}]"
    
    # Replace the original timing with the corrected one
    corrected = re.sub(r'\[.*?\]', time_str, transcription, count=1)
    return corrected

def start_recording(buffer_size_seconds=5, samplerate=16000, model_string='ggml-small.en-tdrz.bin', verbose=1, use_tinydiarize=True, output_file="transcription.txt") -> None:
    buffer_size_frames = int(buffer_size_seconds * samplerate)
    audio_queue = queue.Queue()
    start_time = 0.0

    # Clear the output file if it exists
    open(output_file, 'w').close()

    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    # Start the processing thread
    processing_thread = threading.Thread(target=process_audio_chunk, 
                                         args=(audio_queue, samplerate, model_string, verbose, use_tinydiarize, output_file, start_time))
    processing_thread.start()

    try:
        with sd.InputStream(callback=audio_callback, dtype='int16', channels=1, 
                            samplerate=samplerate, blocksize=buffer_size_frames):
            print(f"Recording... Press Ctrl+C to stop. Output will be saved to {output_file}")
            while True:
                sd.sleep(100)  # Small sleep to prevent CPU overuse
    except KeyboardInterrupt:
        print('Recording stopped.')
    finally:
        # Signal the processing thread to stop
        audio_queue.put(None)
        processing_thread.join()

if __name__ == '__main__':
    model_string = 'ggml-small.en-tdrz.bin'
    buffer_size_seconds = 5 
    verbose = 0
    use_tinydiarize = True
    output_file = "transcription.txt"   
    start_recording(buffer_size_seconds=buffer_size_seconds, model_string=model_string, 
                    verbose=verbose, use_tinydiarize=use_tinydiarize, output_file=output_file)