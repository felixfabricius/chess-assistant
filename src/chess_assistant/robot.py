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
    "You are a funny British chess commentator. I will provide you with context about a chess move and want you to return a one-liner. "
    "Goal: entertainment. If possible, make the comment funny. But don't force it. Sometimes simple, neutral comments might be beter. "
    "References to chess culture (Magnus Carlsen, Hikaru as successful player examples); 'Botez Gambit' when blundering a good piece; "
    "names of openings; are welcome. "
    "I will provide you with a move, the moved piece, the move turn (e.g. if 1st turn: can comment on game starting), whether it was "
    "white or black's turn, the centipawn loss associated with that move "
    "(higher means move was worse; best is zero; anything above 100 is poor, anything above 300 a clear blunder), "
    "and potential extra move indicators (e.g. capture? castle? etc). "
    "I will also provide you with some info about the recent move history within the game: actual recent moves, number of subsequent moves with "
    "(aggressive playing) / without (potential pacifism comment) "
    "capture or average recent centipawn loss (can joke about it being very high (poor chess game), or low)."
    "Additionally, I will provide list of the recent comments in this game, ordered by turn (earlier comments earlier), and by whose "
    "turn it was during that comment. There might be potential for a funny callback; try avoid unintentionally repeating comments / phrases, though.\n"
)
PROMPT_END = (
    "Based on this, return a one-liner (maybe two liner if very strong comment). Applaud solid moves, and roast bad ones. "
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
        self.game_history = {}
        self.comment_history = {"white": [], "black": []}
        self.pregenerated_comments = {}
        self.turn = 0

    def suggest_move(self, move):
        text = f"{format_uci_for_speech(move)}?"
        say(self.mini, self.pipeline, text)

    def pregenerate_comment(self, move: dict, turn: int):
        if turn > self.turn: # reset the comment store for each new turn
            self.pregenerated_comments = {}
        move_uci = move["move"]
        move_info = move["move_info"]
        prompt_move_info = [
            for attribute in [
                "moved_piece", 
            ]
        ]f"{
            
            
        }"

        

    def comment_on_move(
        self, 
        move,
        moved_piece,
        ,
        turn,
        capture,
        captured_piece,
        castle,
        en_passant,
        check,
        checkmate
    ):
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
