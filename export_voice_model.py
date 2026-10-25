"""Convert the emotion2vec+ voice-emotion model to ONNX once, so the app runs it with ONNX Runtime (no PyTorch).

emotion2vec+ is published only as a PyTorch checkpoint for FunASR. This script downloads the pinned checkpoint,
checks its SHA-256, exports it to ONNX, checks that the ONNX model gives the same probabilities as the original on
test signals, and writes it to EMOTION_MODEL_DIR together with a checksum file the app verifies before loading.

Run it in a separate environment (PyTorch + FunASR are large and are not app dependencies):
    python -m venv export-venv
    export-venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cpu
    export-venv\\Scripts\\pip install funasr huggingface_hub onnx onnxruntime
    export-venv\\Scripts\\python export_voice_model.py instance/models
Deploying: run it once at build time (or copy the two output files from a machine that did).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = "emotion2vec/emotion2vec_plus_base"
REVISION = "b318240bfe67db81a8c572ecb37ce9c3759b81c9"
CHECKSUMS = {"model.pt": "60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd",
             "config.yaml": "50674ec187406b4f1208925594ceaf1f28f16a22333b5f2ac5b3d0cbffdd670e",
             "tokens.txt": "866121e470057b847d7a50e9923509141fb2924392f53385a186482a1ec0fb7f"}
NAME = "emotion2vec_plus_base"  # the app's VOICE_EMOTION_MODEL


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(out_dir):
    import torch
    from funasr import AutoModel
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=list(CHECKSUMS)))
    for name, expected in CHECKSUMS.items():
        if sha256(source / name) != expected:
            sys.exit(f"{name} doesn't match its checksum: refusing to convert it")
    labels = [line.split("/")[-1].strip() for line in (source / "tokens.txt").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    auto = AutoModel(model=str(source), disable_update=True, device="cpu")
    model = auto.model.eval()

    class Classifier(torch.nn.Module):
        """FunASR's utterance inference: layer-normalised waveform -> features -> mean over time -> class logits."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, wav):  # (1, samples) float32 at 16 kHz
            # = F.layer_norm(wav, wav.shape) as FunASR does, written out so the length can vary in ONNX
            mean = wav.mean(dim=-1, keepdim=True)
            wav = (wav - mean) / torch.sqrt(((wav - mean) ** 2).mean(dim=-1, keepdim=True) + 1e-5)
            x = self.m.extract_features(wav, padding_mask=None)["x"]
            return self.m.proj(x.mean(dim=1))

    wrapper = Classifier(model).eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{NAME}.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (torch.randn(1, 32000),), str(onnx_path), input_names=["wav"],
                          output_names=["logits"], dynamic_axes={"wav": {1: "samples"}}, opset_version=17,
                          dynamo=False)

    # The ONNX model must match the original on different lengths and kinds of sound
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 7) / 16000
    tests = [rng.normal(0, 0.1, 16000).astype(np.float32), (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
             rng.normal(0, 0.02, 16000 * 20).astype(np.float32)]
    for wav in tests:
        with torch.no_grad():
            expected = wrapper(torch.from_numpy(wav)[None]).numpy()[0]
        got = session.run(None, {"wav": wav[None]})[0][0]
        if not np.allclose(expected, got, atol=1e-3):
            onnx_path.unlink()
            sys.exit(f"The ONNX model differs from the original (max diff {np.abs(expected - got).max():.4f})")

    info = {"source": f"{REPO}@{REVISION}", "labels": labels, "sha256": sha256(onnx_path)}
    (out_dir / f"{NAME}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB) and {NAME}.json; labels: {labels}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python export_voice_model.py <EMOTION_MODEL_DIR>")
    main(sys.argv[1])
