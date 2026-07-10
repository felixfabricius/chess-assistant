import numpy as np
import chess
import anthropic
from scipy.signal import resample
from kokoro import KPipeline
from reachy_mini import ReachyMini

from dotenv import load_dotenv

load_dotenv()

KOKORO_SAMPLE_RATE = 24000

PROMPT_START = (
    "You are a witty, British chess commentator. "
    "I will provide you with a move, the centipawn loss associated with that move (higher means move was worse; best is zero), "
    "and the piece that moved."
    "Based on that, return a SHORT comment. Applaud solid moves, and roast bad ones."
    "Return only the comment."
)
def format_uci_for_speech(uci_move: str) -> str:
    """Turn 'e2e4' into 'E2 to E4' so it's said clearly instead of
    being mangled as one run-together token."""
    origin, destination = uci_move[:2], uci_move[2:4]
    return f"{origin.upper()} to {destination.upper()}"

def synthesize(pipeline, text: str, voice: str = "bm_george", speed: float = 1.0) -> np.ndarray:
    """Run text through Kokoro and return one concatenated float32 waveform."""
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio.numpy() if hasattr(audio, "numpy") else audio)
    return np.concatenate(chunks)

def say(mini: ReachyMini, pipeline, text: str, voice: str = "bm_george"):
    """Synthesize text with Kokoro and play it through Reachy Mini's speaker."""
    audio = synthesize(pipeline, text, voice=voice)

    target_rate = mini.media.get_output_audio_samplerate()
    if target_rate != KOKORO_SAMPLE_RATE:
        audio = resample(audio, int(len(audio) * target_rate / KOKORO_SAMPLE_RATE))

    mini.media.start_playing()
    mini.media.push_audio_sample(audio.astype(np.float32))
    import time
    time.sleep(len(audio) / target_rate)
    mini.media.stop_playing()


class Speaker:
    def __init__(self, mini):
        self.mini = mini
        self.pipeline = KPipeline(lang_code="b")
        self.client = anthropic.Anthropic()

    def suggest_move(self, move):
        text = f"{format_uci_for_speech(move)}?"
        say(self.mini, self.pipeline, text)

    def comment_on_move(self, move, rating, piece):
        prompt = f"{PROMPT_START}\nMove: {move}\nMoved piece: {piece}\nCentipawn loss: {rating}"
        message = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=128,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        comment = message.content[0].text

        say(self.mini, self.pipeline, comment)

    def exclaim_win(self, game):
        winner = "black" if game.board.turn == chess.WHITE else "white"
        say(self.mini, self.pipeline, f"{winner} won!!!")
