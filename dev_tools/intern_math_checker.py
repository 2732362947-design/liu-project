from agents.answer_normalizer import answers_match, normalize_answer


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
        answer_type = item.get("expected_answer_type") or item.get("answer_type")
        normalized_final = normalize_answer(final_answer)
        normalized_expected = normalize_answer(expected_answer)
        if not normalized_final and answer_type != "proof":
            return {
                "status": "unknown",
                "uses_fallback": False,
                "method": "reference_answer",
                "reason": "final_answer is empty",
            }
        matched, reason = answers_match(
            final_answer,
            expected_answer,
            answer_type=answer_type,
            problem=item.get("problem"),
            solution=item.get("solution"),
        )
        if matched:
            return {
                "status": "passed",
                "uses_fallback": False,
                "method": "reference_answer",
                "reason": reason,
                "normalized_final": normalized_final,
                "normalized_expected": normalized_expected,
            }
        return {
            "status": "failed",
            "uses_fallback": False,
            "method": "reference_answer",
            "reason": reason,
            "normalized_final": normalized_final,
            "normalized_expected": normalized_expected,
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
