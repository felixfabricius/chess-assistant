import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import chess
import anthropic
from scipy.signal import resample
from kokoro import KPipeline
from reachy_mini import ReachyMini

from dotenv import load_dotenv

load_dotenv()

KOKORO_SAMPLE_RATE = 24000

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 128
DEFAULT_VOICE = "bm_george"

SYSTEM_PROMPT = (
    "You are a funny British chess commentator. I will provide you with context about a chess move and want you to return a one-liner. "
    "Goal: entertainment. If possible, make the comment funny. But don't force it. Sometimes simple, neutral comments might be better. "
    "References to chess culture (Magnus Carlsen, Hikaru as successful player examples); 'Botez Gambit' when blundering a good piece; "
    "names of openings; are welcome! "
    "I will provide you with a move, the moved piece, the move turn (e.g. if 1st turn: can comment on game starting), whether it was "
    "white or black's turn, the centipawn loss associated with that move "
    "(higher means move was worse; best is zero; anything above 100 is poor, anything above 300 a clear blunder), "
    "and potential extra move indicators (e.g. capture? castle? etc). "
    "I will also provide you with some info about the recent move history within the game: actual recent moves, number of subsequent moves with "
    "(aggressive playing) / without (potential pacifism comment) "
    "capture, and average and recent centipawn loss (can joke about it being very high (poor chess game), or low). "
    "Additionally, I will provide a list of the recent comments in this ga me, ordered by turn (earlier comments earlier), and by whose "
    "turn it was during that comment. There might be potential for a funny callback; try avoid unintentionally repeating comments / phrases, though."
)
PROMPT_END = (
    "Based on this, return a one-liner. Applaud solid moves, and roast bad ones. "
    # Kokoro synthesis runs at roughly 0.6x realtime, so every spoken second costs ~0.6s of
    # synthesis. Keeping the comment short is what keeps it inside the move-review window --
    # and a rambling 'one-liner' is not a one-liner anyway.
    "Hard limit: at most 20 words, one sentence. Be punchy; cut any word that isn't earning its place. "
    "Return only the comment, with no preamble."
)


def format_uci_for_speech(uci_move: str) -> str:
    """Turn 'e2e4' into 'E2 to E4' so it's said clearly instead of
    being mangled as one run-together token."""
    origin, destination = uci_move[:2], uci_move[2:4]
    return f"{origin.upper()} to {destination.upper()}"


def synthesize(pipeline, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> np.ndarray:
    """Run text through Kokoro and return one concatenated float32 waveform."""
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        chunks.append(audio.numpy() if hasattr(audio, "numpy") else audio)
    return np.concatenate(chunks)


def play(mini: ReachyMini, audio: np.ndarray, source_rate: int = KOKORO_SAMPLE_RATE):
    """Play an already-synthesized waveform through Reachy Mini's speaker.

    Kept separate from synthesize() so a worker thread can do the (slow) synthesis
    while playback stays on the main thread -- mini.media must only ever be touched
    from one thread.
    """
    target_rate = mini.media.get_output_audio_samplerate()
    if target_rate != source_rate:
        audio = resample(audio, int(len(audio) * target_rate / source_rate))

    mini.media.start_playing()
    mini.media.push_audio_sample(audio.astype(np.float32))
    time.sleep(len(audio) / target_rate)
    mini.media.stop_playing()


def say(mini: ReachyMini, pipeline, text: str, voice: str = DEFAULT_VOICE):
    """Synthesize text with Kokoro and play it through Reachy Mini's speaker."""
    play(mini, synthesize(pipeline, text, voice=voice))


def _yes_no(value) -> str:
    return "yes" if value else "no"


def build_prompt(move_info: dict, cp_loss: int, history: dict, comment_history: list) -> str:
    """Assemble the user turn: just the facts. The instructions live in SYSTEM_PROMPT."""
    capture = "no"
    if move_info["capture"]:
        captured = move_info.get("captured_piece") or "a piece"
        capture = f"yes - took a {captured}"
        if move_info["en_passant"]:
            capture += " (en passant)"

    lines = [
        f"Move: {move_info['san']} ({move_info['move']})",
        f"Moved piece: {move_info['moved_piece']}",
        f"Turn: {move_info['move_number']}, {move_info['turn']} to move",
        f"Centipawn loss: {cp_loss}",
        f"Capture: {capture}",
        f"Castle: {move_info['castle'] or 'no'}",
        f"En passant: {_yes_no(move_info['en_passant'])}",
        f"Promotion: {move_info['promotion'] or 'no'}",
        f"Check: {_yes_no(move_info['check'])}",
        f"Checkmate: {_yes_no(move_info['checkmate'])}",
    ]

    recent_moves = history["recent_moves"]
    averages = history["average_cp_loss"]
    last_losses = history["last_cp_losses"]
    lines += [
        "",
        "Game history:",
        f"Recent moves: {' '.join(recent_moves) if recent_moves else '(none - this is the first move)'}",
        f"Consecutive captures: {history['capture_streak']}",
        f"Consecutive quiet moves: {history['quiet_streak']}",
        f"Average centipawn loss - white: {averages['white']:.1f}, black: {averages['black']:.1f}",
        f"Last centipawn losses - white: {last_losses['white']}, black: {last_losses['black']}",
    ]

    lines += ["", "Comment history:"]
    if comment_history:
        for entry in comment_history:
            lines.append(
                f"Turn {entry['turn']}; {entry['side']}; {entry['move']}; \"{entry['comment']}\""
            )
    else:
        lines.append("(none yet - this is the first comment of the game)")

    lines += ["", PROMPT_END]
    return "\n".join(lines)


class Speaker:
    def __init__(self, mini, config=None):
        speaker_config = config.get("speaker", {}) if config is not None else {}

        self.mini = mini
        self.pipeline = KPipeline(lang_code="b")
        self.client = anthropic.Anthropic()

        self.model = speaker_config.get("model", DEFAULT_MODEL)
        self.max_tokens = speaker_config.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.voice = speaker_config.get("voice", DEFAULT_VOICE)
        self.n_recent_moves = speaker_config.get("recent_moves", 6)
        self.n_recent_cp_losses = speaker_config.get("recent_cp_losses", 5)

        # Turn-ordered so the "Turn; side; move; comment" lines the prompt asks for fall
        # straight out of it.
        self.comment_history = []

        # move_uci -> Future[{"comment", "audio", "cp_loss", "new_score"}], populated
        # during the move-review window and consumed the moment a move is accepted.
        self.pregenerated_comments = {}
        self.turn = 0

        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="pregen")
        # KPipeline is a torch model with no thread-safety guarantees, and the main thread
        # still synthesizes the "E2 to E4?" prompts while a worker synthesizes a comment.
        self.pipeline_lock = threading.Lock()

    def _synthesize(self, text: str) -> np.ndarray:
        with self.pipeline_lock:
            return synthesize(self.pipeline, text, voice=self.voice)

    def suggest_move(self, move):
        text = f"{format_uci_for_speech(move)}?"
        play(self.mini, self._synthesize(text))

    def pregenerate_comment(self, move: dict, turn: int, game) -> None:
        """Kick off comment generation for a candidate move and return immediately.

        Called right after the candidate is spoken aloud, so the engine analysis, the
        Claude call and the Kokoro synthesis all happen while the players are using the
        review window to accept or reject the suggestion.
        """
        if turn > self.turn:  # new turn -> last turn's candidates are dead
            self.pregenerated_comments = {}
            self.turn = turn

        move_uci = move["move"]
        if move_uci in self.pregenerated_comments:
            return  # already generated (or generating) for this candidate

        move_info = move["move_info"]
        # Snapshot on the main thread: the worker must not read move_log / comment_history
        # while a later move is being appended to them.
        history = game.history_snapshot(
            recent_moves=self.n_recent_moves,
            recent_cp_losses=self.n_recent_cp_losses,
        )
        comment_history = list(self.comment_history)

        self.pregenerated_comments[move_uci] = self.executor.submit(
            self._generate, game, move_uci, move_info, history, comment_history
        )

    def _generate(self, game, move_uci, move_info, history, comment_history):
        """Runs on a worker thread. Never touches mini.media."""
        cp_loss, new_score = game.cp_loss_for(move_uci)

        prompt = build_prompt(move_info, cp_loss, history, comment_history)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        comment = next(block.text for block in message.content if block.type == "text")

        return {
            "comment": comment,
            "audio": self._synthesize(comment),
            "cp_loss": cp_loss,
            "new_score": new_score,
        }

    def comment_on_move(self, move_uci: str, move_info: dict, game):
        """Speak the comment for an accepted move and record it.

        Returns (cp_loss, new_score) so the caller can pass them to game.apply_move()
        rather than paying for a second Stockfish analysis of the same position.
        """
        future = self.pregenerated_comments.get(move_uci)
        if future is None:
            # Never pregenerated (shouldn't normally happen) - generate inline.
            history = game.history_snapshot(
                recent_moves=self.n_recent_moves,
                recent_cp_losses=self.n_recent_cp_losses,
            )
            result = self._generate(
                game, move_uci, move_info, history, list(self.comment_history)
            )
        else:
            # Blocks only if the worker hasn't finished yet, i.e. no worse than the old
            # fully-synchronous path.
            result = future.result()

        play(self.mini, result["audio"])

        self.comment_history.append({
            "turn": move_info["move_number"],
            "side": move_info["turn"],
            "move": move_info["move"],
            "comment": result["comment"],
        })

        return result["cp_loss"], result["new_score"]

    def exclaim_win(self, game):
        winner = "black" if game.board.turn == chess.WHITE else "white"
        play(self.mini, self._synthesize(f"{winner} won!!!"))

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
