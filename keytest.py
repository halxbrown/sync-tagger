"""Which Essentia key profile matches published keys best on YOUR songs?

Usage (inside the container, with your samples folder mounted at /audio):
    python keytest.py /audio/keys.csv

keys.csv: one song per line,  filename,published key   e.g.
    First Date.mp3,C major
    Curls.mp3,G minor
Use # for sharps and b for flats (C# minor, Ab major).
"""
import csv
import sys

import essentia
from essentia.standard import KeyExtractor, MonoLoader

essentia.log.infoActive = False
essentia.log.warningActive = False

PROFILES = ["bgate", "edma", "edmm", "krumhansl", "temperley", "temperley2005",
            "shaath", "noland", "diatonic", "tonictriad", "gomez", "braw"]
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}


def parse(key: str):
    root, scale = key.strip().replace("♯", "#").replace("♭", "b").split()
    root = FLATS.get(root[0].upper() + root[1:], root[0].upper() + root[1:])
    return NOTES.index(root), scale.lower()


def camelot(pc: int, scale: str):
    """Camelot wheel position (1-12) and letter: A = minor, B = major."""
    major_pc = pc if scale == "major" else (pc + 3) % 12        # relative major
    return ((major_pc * 7) % 12 + 7) % 12 + 1, "B" if scale == "major" else "A"


def relation(pub, det):
    if pub == det:
        return "exact"
    (n1, l1), (n2, l2) = camelot(*pub), camelot(*det)
    step = min((n1 - n2) % 12, (n2 - n1) % 12)
    if step == 0:
        return "relative"           # same notes: C major / A minor
    if step == 1:
        return "neighbor"           # one step on the wheel (fifth apart, or diagonal)
    return "off"


def main(csv_path: str):
    folder = csv_path.rsplit("/", 1)[0]
    rows = [r for r in csv.reader(open(csv_path, encoding="utf-8-sig")) if r and not r[0].startswith("#")]
    score = {p: {"exact": 0, "relative": 0, "neighbor": 0, "off": 0} for p in PROFILES}

    for row in rows:
        fname, pub_key = row[0], ",".join(row[1:]).split("#tunebat")[0]
        pub_key = pub_key.split(" # ")[0]              # allow notes like "C major # tunebat"
        audio = MonoLoader(filename=f"{folder}/{fname.strip()}", sampleRate=44100)()
        pub = parse(pub_key)
        line = []
        for p in PROFILES:
            k, s, _ = KeyExtractor(profileType=p)(audio)
            det = (NOTES.index(FLATS.get(k, k)), s)
            rel = relation(pub, det)
            score[p][rel] += 1
            line.append(f"{p}={k} {s}" + ("" if rel == "exact" else f" ({rel})"))
        print(f"\n{fname}  [published: {pub_key}]\n   " + "\n   ".join(line))

    n = len(rows)
    print(f"\n=== {n} songs ===   exact | +relative | +neighbor")
    ranked = sorted(PROFILES, key=lambda p: (score[p]["exact"], score[p]["relative"], score[p]["neighbor"]), reverse=True)
    for p in ranked:
        s = score[p]
        e, r, nb = s["exact"], s["exact"] + s["relative"], s["exact"] + s["relative"] + s["neighbor"]
        print(f"{p:<14} {e:>2}/{n}   {r:>2}/{n}   {nb:>2}/{n}")


if __name__ == "__main__":
    main(sys.argv[1])
