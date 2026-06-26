from agents.answer_normalizer import normalize_answer


def test_fraction_normalization():
    assert normalize_answer(r"\frac{3}{5}") == normalize_answer("3/5")


def test_root_set_normalization():
    assert normalize_answer("x_1=2, x_2=3") == normalize_answer("{2,3}")


def test_none_normalization():
    assert normalize_answer(None) == ""
