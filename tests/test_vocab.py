from ecgcode import vocab


def test_id_count():
    assert len(vocab.ID_TO_CHAR) == 13


def test_pad_is_zero():
    assert vocab.ID_PAD == 0
    assert vocab.ID_TO_CHAR[0] == "·"


def test_active_v1_classes_are_nine():
    assert len(vocab.ACTIVE_V1) == 9


def test_active_set_is_correct():
    expected = {vocab.ID_UNK, vocab.ID_ISO, vocab.ID_PACER,
                vocab.ID_P, vocab.ID_Q, vocab.ID_R, vocab.ID_S,
                vocab.ID_T, vocab.ID_W}
    assert set(vocab.ACTIVE_V1) == expected


def test_char_lookup_roundtrip():
    for i, ch in vocab.ID_TO_CHAR.items():
        assert vocab.CHAR_TO_ID[ch] == i


def test_name_lookup_roundtrip():
    for i, name in vocab.ID_TO_NAME.items():
        assert vocab.NAME_TO_ID[name] == i


def test_no_duplicate_chars():
    chars = list(vocab.ID_TO_CHAR.values())
    assert len(chars) == len(set(chars))


def test_no_duplicate_names():
    names = list(vocab.ID_TO_NAME.values())
    assert len(names) == len(set(names))


def test_pacer_id_is_3():
    assert vocab.ID_PACER == 3


def test_iso_id_is_2():
    assert vocab.ID_ISO == 2


def test_w_id_is_10():
    assert vocab.ID_W == 10
