"""Sync-licensing auto-tagger: audio file -> genre / mood / instrument tags, vocals, BPM, key.

Pipeline (identical to training in Colab):
  audio @ 16 kHz -> three 10 s chunks (25/50/75% of the track) -> mel patches
  -> Discogs-EffNet (ONNX) -> per-patch 1280-d embeddings -> mean + std = 2560-d
  -> trained head (ONNX, normalization + sigmoid built in) -> per-tag probabilities
  -> per-tag thresholds (tuned on validation) -> tags

Extras:
  - vocals: Essentia's pretrained voice/instrumental head, converted to ONNX, run on
    the same per-patch embeddings (optional: skipped if the model file is missing)
  - BPM and key: Essentia's signal-processing extractors, with alternate readings
"""
import json
import os

import numpy as np
import onnxruntime as ort
import essentia
from essentia.standard import (
    FrameGenerator,
    KeyExtractor,
    MonoLoader,
    RhythmExtractor2013,
    TensorflowInputMusiCNN,
)

essentia.log.infoActive = False  # keep container logs clean

SR = 16000            # sample rate the backbone was trained on
CHUNK = 10 * SR       # 10-second chunks
PATCH, HOP = 128, 62  # 128 mel frames per patch, hop of 62 frames (Essentia's default)
NEAR_MISS = 0.6       # a tag scoring >= 60% of its threshold is offered as a suggestion

NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
ALIASES = {"D#": "Eb", "G#": "Ab", "A#": "Bb", "Db": "C#", "Gb": "F#"}


def key_info(key: str, scale: str) -> dict:
    """Relative key and Camelot wheel position (A = minor, B = major)."""
    pc = NOTES.index(ALIASES.get(key, key))
    if scale == "major":
        rel, rel_scale, major_pc = NOTES[(pc + 9) % 12], "minor", pc
    else:
        rel, rel_scale, major_pc = NOTES[(pc + 3) % 12], "major", (pc + 3) % 12
    number = ((major_pc * 7) % 12 + 7) % 12 + 1
    return {"key": f"{NOTES[pc]} {scale}", "key_relative": f"{rel} {rel_scale}",
            "camelot": f"{number}{'B' if scale == 'major' else 'A'}"}


def tempo_alternates(bpm: float) -> list:
    """Other plausible readings of the same pulse: half/double time and 3:2 feels."""
    cands = [(bpm / 2, "half-time"), (bpm * 2, "double-time"),
             (bpm * 1.5, "x1.5"), (bpm * 2 / 3, "x2/3")]
    return [{"bpm": round(b, 1), "label": l} for b, l in cands if 55 <= b <= 210]


class Tagger:
    def __init__(self, model_dir: str = "models"):
        cpu = ["CPUExecutionProvider"]
        self.backbone = ort.InferenceSession(f"{model_dir}/discogs-effnet-bsdynamic-1.onnx", providers=cpu)
        self.head = ort.InferenceSession(f"{model_dir}/tagger_head_v2.onnx", providers=cpu)

        vi_path = f"{model_dir}/voice_instrumental-discogs-effnet-1.onnx"
        self.voice = ort.InferenceSession(vi_path, providers=cpu) if os.path.exists(vi_path) else None
        self.voice_input = self.voice.get_inputs()[0].name if self.voice else None

        with open(f"{model_dir}/tagger_meta.json") as f:
            meta = json.load(f)
        self.heads = meta["heads"]                                   # ["genre", "mood_theme", "instrument"]
        self.tags = meta["tags"]                                     # tag names, in output order
        self.thresholds = {h: np.array(v) for h, v in meta["thresholds"].items()}

        self.mel = TensorflowInputMusiCNN()

    # ---------- audio -> embedding ----------
    def _patches(self, audio: np.ndarray) -> np.ndarray:
        # startFromZero=False matches Essentia's internal framing (verified: cosine 1.0000 vs bs64 model)
        frames = FrameGenerator(audio, frameSize=512, hopSize=256, startFromZero=False)
        mel = np.array([self.mel(fr) for fr in frames], dtype=np.float32)   # (frames, 96)
        n = (len(mel) - PATCH) // HOP + 1
        if n < 1:
            return np.zeros((0, PATCH, 96), np.float32)
        return np.stack([mel[i * HOP: i * HOP + PATCH] for i in range(n)])

    @staticmethod
    def _windows(n_samples: int) -> list:
        """(start, end) sample indices of the audio the model listens to."""
        if n_samples <= 3 * CHUNK:           # short track: use all of it
            return [(0, n_samples)]
        mids = [int(f * n_samples) for f in (0.25, 0.5, 0.75)]
        return [(m - CHUNK // 2, m + CHUNK // 2) for m in mids]

    def patch_embeddings(self, audio: np.ndarray) -> np.ndarray:
        # patches are built per chunk, so no patch straddles two unrelated moments
        chunks = [audio[s:e] for s, e in self._windows(len(audio))]
        patches = np.concatenate([self._patches(c) for c in chunks])
        if len(patches) == 0:
            raise ValueError("Audio is too short to analyze (need at least ~2 seconds).")
        return self.backbone.run(["embeddings"], {"melspectrogram": patches})[0]   # (patches, 1280)

    # ---------- deterministic features ----------
    @staticmethod
    def _music_info(path: str) -> dict:
        audio = MonoLoader(filename=path, sampleRate=44100)()   # these extractors expect 44.1 kHz
        bpm, _, beat_conf, _, _ = RhythmExtractor2013(method="multifeature")(audio)
        key, scale, key_strength = KeyExtractor()(audio)
        # tempo detectors often land on half-time, double-time or a 3:2 feel; offer the other
        # readings instead of silently "correcting" (a real 155 BPM beat must stay 155)
        alts = tempo_alternates(float(bpm))
        return {
            "bpm": round(float(bpm), 1),
            "bpm_alt": alts[0]["bpm"] if alts else round(float(bpm), 1),
            "bpm_alts": alts,
            "bpm_confidence": round(float(beat_conf), 2),
            **key_info(key, scale),
            "key_strength": round(float(key_strength), 2),
            "duration_sec": round(len(audio) / 44100, 1),
        }

    # ---------- public API ----------
    def predict(self, path: str) -> dict:
        audio = MonoLoader(filename=path, sampleRate=SR, resampleQuality=4)()
        patch_emb = self.patch_embeddings(audio)
        emb = np.concatenate([patch_emb.mean(0), patch_emb.std(0)]).astype(np.float32)   # (2560,)
        probs = self.head.run(None, {"embedding": emb[None]})    # one array per head, in self.heads order

        result, suggestions = {}, {}
        for h, p in zip(self.heads, probs):
            p, th = p[0], self.thresholds[h]
            hits = [{"tag": t, "score": round(float(s), 3)}
                    for t, s, c in zip(self.tags[h], p, th) if s >= c]
            near = [{"tag": t, "score": round(float(s), 3), "of_threshold": round(float(s / c), 2)}
                    for t, s, c in zip(self.tags[h], p, th) if NEAR_MISS * c <= s < c]
            result[h] = sorted(hits, key=lambda x: -x["score"])
            suggestions[h] = sorted(near, key=lambda x: -x["of_threshold"])[:3]
        result["suggestions"] = suggestions

        # vocals: average the per-patch [instrumental, voice] probabilities
        if self.voice is not None:
            x = patch_emb.astype(np.float32)
            if self.voice.get_inputs()[0].shape[0] == 1:          # converted with a fixed batch of 1
                out = np.concatenate([self.voice.run(None, {self.voice_input: x[i:i + 1]})[0] for i in range(len(x))])
            else:
                out = self.voice.run(None, {self.voice_input: x})[0]
            v = float(out[:, 1].mean())
            result["vocals"] = {"voice": round(v, 3), "instrumental": v < 0.5}
        else:
            result["vocals"] = None

        # where in the track the model listened, in seconds (shown on the waveform)
        result["windows"] = [[round(s / SR, 2), round(e / SR, 2)] for s, e in self._windows(len(audio))]
        result.update(self._music_info(path))
        return result


if __name__ == "__main__":
    import sys
    print(json.dumps(Tagger().predict(sys.argv[1]), indent=2))
