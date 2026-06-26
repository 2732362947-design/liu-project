from agents.answer_normalizer import normalize_answer


def check_math_result(item: dict) -> dict:
    if item.get("fallback_final_answer"):
        return {
            "status": "failed",
            "uses_fallback": True,
            "method": "fallback_guard",
            "reason": "fallback answers are not accepted as formal math checks",
        }

    expected_answer = item.get("expected_answer")
    final_answer = item.get("final_answer")
    if expected_answer:
        normalized_final = normalize_answer(final_answer)
        normalized_expected = normalize_answer(expected_answer)
        if not normalized_final:
            return {
                "status": "unknown",
                "uses_fallback": False,
                "method": "reference_answer",
                "reason": "final_answer is empty",
            }
        if normalized_final == normalized_expected:
            return {
                "status": "passed",
                "uses_fallback": False,
                "method": "reference_answer",
                "reason": "final_answer matches expected_answer",
            }
        return {
            "status": "failed",
            "uses_fallback": False,
            "method": "reference_answer",
            "reason": "final_answer does not match expected_answer",
        }

    return {
        "status": "unknown",
        "uses_fallback": False,
        "method": "none",
        "reason": "expected_answer is not available",
    }


def check_math_result_from_parts(problem: str, solution: str, final_answer: str | None = None) -> dict:
    return check_math_result(
        {
            "problem": problem,
            "solution": solution,
            "final_answer": final_answer,
        }
    )
