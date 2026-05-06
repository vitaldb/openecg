"""OpenECG alphabet v1.0 — 13 IDs, append-only versioning.

See spec: docs/superpowers/specs/2026-05-03-ecgcode-stage1-design.md
"""

VOCAB_VERSION = "1.0"

ID_PAD = 0
ID_UNK = 1
ID_ISO = 2
ID_PACER = 3
ID_P = 4
ID_Q = 5
ID_R = 6
ID_S = 7
ID_T = 8
ID_U = 9
ID_W = 10
ID_D = 11
ID_J = 12

ID_TO_CHAR = {
    ID_PAD: "·",
    ID_UNK: "?",
    ID_ISO: "_",
    ID_PACER: "*",
    ID_P: "p",
    ID_Q: "q",
    ID_R: "r",
    ID_S: "s",
    ID_T: "t",
    ID_U: "u",
    ID_W: "w",
    ID_D: "d",
    ID_J: "j",
}

ID_TO_NAME = {
    ID_PAD: "<pad>",
    ID_UNK: "<unk>",
    ID_ISO: "iso",
    ID_PACER: "pacer_spike",
    ID_P: "P",
    ID_Q: "Q",
    ID_R: "R",
    ID_S: "S",
    ID_T: "T",
    ID_U: "U",
    ID_W: "wide_QRS",
    ID_D: "delta",
    ID_J: "J_wave",
}

CHAR_TO_ID = {ch: i for i, ch in ID_TO_CHAR.items()}
NAME_TO_ID = {name: i for i, name in ID_TO_NAME.items()}

# Active classes in v1.0 (predicted by model + emitted by labeler).
# Excludes pad (mask only) and reserved IDs (u, d, j — v1.1).
ACTIVE_V1 = (
    ID_UNK, ID_ISO, ID_PACER,
    ID_P, ID_Q, ID_R, ID_S, ID_T, ID_W,
)
