from output import safe_divide

def test_divide():
    assert safe_divide(10, 2) == 5
    assert safe_divide(5, 0) == "error"