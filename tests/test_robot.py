import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from chess_assistant import robot
from chess_assistant.robot import PROMPT_END, Speaker, build_prompt, format_uci_for_speech


def move_info(**overrides):
    base = {
        "move": "f3f7",
        "san": "Qxf7#",
        "moved_piece": "Queen",
        "turn": "white",
        "move_number": 4,
        "capture": True,
        "captured_piece": "Pawn",
        "castle": None,
        "en_passant": False,
        "promotion": None,
        "check": True,
        "checkmate": True,
    }
    return {**base, **overrides}


def history(**overrides):
    base = {
        "recent_moves": ["e2e4", "e7e5", "f1c4", "b8c6", "d1f3"],
        "capture_streak": 0,
        "quiet_streak": 5,
        "average_cp_loss": {"white": 12.5, "black": 88.0},
        "last_cp_losses": {"white": [0, 25], "black": [4, 172]},
    }
    return {**base, **overrides}


def test_prompt_contains_every_move_field():
    prompt = build_prompt(move_info(), cp_loss=0, history=history(), comment_history=[])

    assert "Move: Qxf7# (f3f7)" in prompt
    assert "Moved piece: Queen" in prompt
    assert "Turn: 4, white to move" in prompt
    assert "Centipawn loss: 0" in prompt
    assert "Capture: yes - took a Pawn" in prompt
    assert "Castle: no" in prompt
    assert "Check: yes" in prompt
    assert "Checkmate: yes" in prompt
    assert prompt.rstrip().endswith(PROMPT_END)


def test_prompt_contains_game_history():
    prompt = build_prompt(move_info(), cp_loss=240, history=history(), comment_history=[])

    assert "Recent moves: e2e4 e7e5 f1c4 b8c6 d1f3" in prompt
    assert "Consecutive captures: 0" in prompt
    assert "Consecutive quiet moves: 5" in prompt
    assert "Average centipawn loss - white: 12.5, black: 88.0" in prompt
    assert "Last centipawn losses - white: [0, 25], black: [4, 172]" in prompt


def test_prompt_renders_comment_history_in_turn_order():
    comments = [
        {"turn": 1, "side": "white", "move": "e2e4", "comment": "A bold start."},
        {"turn": 2, "side": "black", "move": "e7e5", "comment": "Mirror, mirror."},
    ]
    prompt = build_prompt(move_info(), cp_loss=0, history=history(), comment_history=comments)

    first = prompt.index('Turn 1; white; e2e4; "A bold start."')
    second = prompt.index('Turn 2; black; e7e5; "Mirror, mirror."')
    assert first < second


def test_castle_and_promotion_render_as_words_not_booleans():
    prompt = build_prompt(
        move_info(castle="queenside", promotion="Knight", capture=False, captured_piece=None),
        cp_loss=15,
        history=history(),
        comment_history=[],
    )
    assert "Castle: queenside" in prompt
    assert "Promotion: Knight" in prompt
    assert "Capture: no" in prompt


def test_en_passant_capture_is_spelled_out():
    prompt = build_prompt(
        move_info(capture=True, captured_piece="Pawn", en_passant=True),
        cp_loss=0,
        history=history(),
        comment_history=[],
    )
    assert "Capture: yes - took a Pawn (en passant)" in prompt
    assert "En passant: yes" in prompt


def test_empty_history_does_not_leak_none_into_the_prompt():
    """First move of the game: every history section is empty. The model should see
    an explanation, not a bare 'None' or an empty line."""
    empty = history(
        recent_moves=[],
        capture_streak=0,
        quiet_streak=0,
        average_cp_loss={"white": 0.0, "black": 0.0},
        last_cp_losses={"white": [], "black": []},
    )
    prompt = build_prompt(
        move_info(move_number=1, capture=False, captured_piece=None, check=False, checkmate=False),
        cp_loss=0,
        history=empty,
        comment_history=[],
    )

    assert "None" not in prompt
    assert "this is the first move" in prompt
    assert "this is the first comment of the game" in prompt


@pytest.mark.parametrize("uci, spoken", [("e2e4", "E2 to E4"), ("g1f3", "G1 to F3")])
def test_format_uci_for_speech(uci, spoken):
    assert format_uci_for_speech(uci) == spoken


### Pregeneration plumbing, exercised offline (no Claude, no Kokoro, no robot)

class FakeClient:
    """Stands in for anthropic.Anthropic(). Records the prompts it was asked to complete."""

    def __init__(self):
        self.prompts = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, system, messages):
        self.prompts.append(messages[0]["content"])
        text = f"comment #{len(self.prompts)}"
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeGame:
    """Stands in for ChessGame. cp_loss_for() is the only thing the worker calls on it."""

    def __init__(self):
        self.rated = []

    def cp_loss_for(self, move_uci):
        self.rated.append(move_uci)
        return 120, -40

    def history_snapshot(self, recent_moves=6, recent_cp_losses=5):
        return {
            "recent_moves": ["e2e4"],
            "capture_streak": 0,
            "quiet_streak": 1,
            "average_cp_loss": {"white": 0.0, "black": 0.0},
            "last_cp_losses": {"white": [], "black": []},
        }


class FakeSpeaker(Speaker):
    """Speaker with the two slow, external dependencies swapped out. Everything else --
    the executor, the cache, the turn reset, the history bookkeeping -- is the real thing."""

    def __init__(self):
        self.mini = None
        self.client = FakeClient()
        self.model = "claude-haiku-4-5"
        self.max_tokens = 64
        self.voice = "bm_george"
        self.n_recent_moves = 6
        self.n_recent_cp_losses = 5
        self.comment_history = []
        self.pregenerated_comments = {}
        self.turn = 0
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.pipeline_lock = threading.Lock()

    def _synthesize(self, text):
        return np.zeros(16, dtype=np.float32)


def candidate(uci, **overrides):
    info = {
        "move": uci, "san": "Nf3", "moved_piece": "Knight", "turn": "white",
        "move_number": 3, "capture": False, "captured_piece": None, "castle": None,
        "en_passant": False, "promotion": None, "check": False, "checkmate": False,
    }
    info.update(overrides)
    return {"move": uci, "loss": 0.1, "move_info": info}


@pytest.fixture
def speaker(monkeypatch):
    # comment_on_move plays through mini.media, which we don't have.
    monkeypatch.setattr(robot, "play", lambda *args, **kwargs: None)
    spk = FakeSpeaker()
    yield spk
    spk.shutdown()


def test_pregenerate_is_cached_per_candidate(speaker):
    """A candidate re-suggested within the same turn must not be regenerated."""
    game = FakeGame()
    speaker.pregenerate_comment(candidate("g1f3"), turn=1, game=game)
    speaker.pregenerate_comment(candidate("g1f3"), turn=1, game=game)

    speaker.pregenerated_comments["g1f3"].result()
    assert len(speaker.client.prompts) == 1
    assert game.rated == ["g1f3"]


def test_pregenerate_cache_resets_on_a_new_turn(speaker):
    """self.turn must actually advance. It never did, so `turn > self.turn` was always
    true and the cache was wiped on every single candidate -- defeating itself."""
    game = FakeGame()
    speaker.pregenerate_comment(candidate("g1f3"), turn=1, game=game)
    speaker.pregenerate_comment(candidate("b1c3"), turn=1, game=game)
    assert speaker.turn == 1
    assert set(speaker.pregenerated_comments) == {"g1f3", "b1c3"}

    speaker.pregenerate_comment(candidate("e2e4"), turn=2, game=game)
    assert speaker.turn == 2
    assert set(speaker.pregenerated_comments) == {"e2e4"}  # last turn's candidates dropped


def test_comment_on_move_returns_the_rating_the_worker_computed(speaker):
    """main.py hands these straight to apply_move, so the accepted move is never
    re-analysed by Stockfish on the critical path."""
    game = FakeGame()
    move = candidate("g1f3")
    speaker.pregenerate_comment(move, turn=1, game=game)

    cp_loss, new_score = speaker.comment_on_move("g1f3", move["move_info"], game)

    assert (cp_loss, new_score) == (120, -40)
    assert game.rated == ["g1f3"]  # rated once, by the worker -- not again here


def test_comment_on_move_records_comment_history(speaker):
    game = FakeGame()
    move = candidate("g1f3")
    speaker.pregenerate_comment(move, turn=1, game=game)
    speaker.comment_on_move("g1f3", move["move_info"], game)

    assert speaker.comment_history == [
        {"turn": 3, "side": "white", "move": "g1f3", "comment": "comment #1"}
    ]

    # ...and that history is fed to the next comment, so callbacks are possible.
    nxt = candidate("b8c6", turn="black", move_number=3)
    speaker.pregenerate_comment(nxt, turn=2, game=game)
    speaker.comment_on_move("b8c6", nxt["move_info"], game)

    assert 'Turn 3; white; g1f3; "comment #1"' in speaker.client.prompts[1]


def test_comment_on_move_generates_inline_if_never_pregenerated(speaker):
    """Fallback path: a move that was never suggested (so never pregenerated) must still
    get a comment rather than blowing up on a missing cache entry."""
    game = FakeGame()
    move = candidate("g1f3")

    cp_loss, _ = speaker.comment_on_move("g1f3", move["move_info"], game)

    assert cp_loss == 120
    assert len(speaker.client.prompts) == 1
    assert speaker.comment_history[0]["comment"] == "comment #1"
