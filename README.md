# chess-assistant

A [Reachy Mini](https://huggingface.co/blog/reachy-mini) desktop robot that watches two humans play
physical chess, reads the board from a single camera photo with a custom 1.3 MB CNN, ranks the legal
moves to work out what you just played, and then comments on it out loud — praising the good moves
and roasting the bad ones.

> **Note:** this README is a work in progress. Setup instructions, architecture, and results are
> still to come.

## License

This project ships three things, and they are **not** under the same license.

| Artifact | License |
| --- | --- |
| Source code | [GPL-3.0-or-later](LICENSE) |
| Bundled model weights (`weights/`) | [Apache-2.0](weights/LICENSE) |
| Training dataset (on Hugging Face, not in this repo) | CC-BY-4.0 |

### Why GPL, and not something more permissive

Not by preference — by obligation. This project depends on
[python-chess](https://github.com/niklasf/python-chess), which is **GPL-3.0-or-later**, and it is
not an incidental dependency: `chess.Board.legal_moves` is what lets the robot rank candidate moves
against a noisy board reading, which is the core idea of the whole system. Distributing a program
that links a GPL library means the combined work is GPL, so GPL-3.0-or-later it is.

Every other dependency is GPL-3.0-compatible: BSD (PyTorch, torchvision, SciPy, OmegaConf),
Apache-2.0 (OpenCV, safetensors, Kokoro), MIT (anthropic, Polars, W&B, Hydra), PSF (Matplotlib),
and LGPL (pygame).

### Stockfish

[Stockfish](https://stockfishchess.org/) is also GPL-3.0, but it imposes nothing here: it is
invoked as a **separate process** over the UCI protocol and its binary is never redistributed with
this repository — you install it yourself. That is an arms-length arrangement, no different from
shelling out to any other program.

### Why the weights are Apache-2.0 rather than GPL

The model weights are the *output* of the training code, not a derivative work of it — the same
reason a program compiled with GCC does not inherit GCC's license, which the FSF
[states explicitly](https://www.gnu.org/licenses/gpl-faq.html#CanIUseGPLToolsForNF). The GPL on the
training code therefore does not reach them, and they are released permissively so that anyone can
reuse them. The same weights and license are published on Hugging Face.

Copyright © 2026 Felix Fabricius.
