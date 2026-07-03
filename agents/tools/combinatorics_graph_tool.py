from __future__ import annotations

import re


def normalize_problem_text(problem: str) -> str:
    text = str(problem or "").lower()
    replacements = {
        r"\{": "{",
        r"\}": "}",
        r"\dots": "...",
        r"\ldots": "...",
        r"\mid": " divides ",
        r"\plus{}": "+",
        r"\plus": "+",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("$", " ")
    text = text.replace("{ }", "{}")
    text = re.sub(r"\s*([,+{}|])\s*", r"\1", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_problem(problem: str) -> str:
    return normalize_problem_text(problem)


def _extract_set_size(text: str) -> int | None:
    match = re.search(r"\{1,2,(?:\.\.\.|…),(\d+)\}", text)
    if match:
        return int(match.group(1))
    return None


def detect_divisibility_subset_problem(problem: str) -> dict | None:
    text = _normalize_problem(problem)
    n = _extract_set_size(text)
    if n is None:
        return None

    required_patterns = (
        r"smallest positive integer\s+k\b",
        r"\bk-element subset\b",
        r"contains two distinct elements",
    )
    if not all(re.search(pattern, text) for pattern in required_patterns):
        return None

    divisibility_patterns = (
        r"a\+b\s+divides\s+ab",
        r"a\+b\|ab",
    )
    if not any(re.search(pattern, text) for pattern in divisibility_patterns):
        return None

    return {"type": "divisibility_subset_graph", "n": n}


def build_divisibility_graph(n: int) -> dict[int, set[int]]:
    if n < 1:
        raise ValueError("n must be positive")
    graph = {value: set() for value in range(1, n + 1)}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                graph[a].add(b)
                graph[b].add(a)
    return graph


def _greedy_color_order(candidates: int, adjacency_masks: list[int]) -> tuple[list[int], list[int]]:
    order: list[int] = []
    colors: list[int] = []
    uncolored = candidates
    color = 0
    while uncolored:
        color += 1
        available = uncolored
        while available:
            bit = available & -available
            vertex = bit.bit_length() - 1
            order.append(vertex)
            colors.append(color)
            uncolored &= ~bit
            available &= ~bit
            available &= ~adjacency_masks[vertex]
    return order, colors


def _vertices_from_mask(mask: int) -> list[int]:
    vertices = []
    while mask:
        bit = mask & -mask
        vertices.append(bit.bit_length())
        mask &= ~bit
    return vertices


def is_independent_set(vertices: list[int], graph: dict[int, set[int]]) -> bool:
    seen = set()
    for vertex in vertices:
        if vertex in seen:
            return False
        seen.add(vertex)
        if any(neighbor in seen for neighbor in graph.get(vertex, set())):
            return False
    return True


def _maximum_clique(adjacency_masks: list[int], n: int) -> tuple[int, list[int]]:
    best = 0
    best_mask = 0

    def expand(size: int, clique_mask: int, candidates: int) -> None:
        nonlocal best, best_mask
        if not candidates:
            if size > best:
                best = size
                best_mask = clique_mask
            return

        order, colors = _greedy_color_order(candidates, adjacency_masks)
        while order:
            if size + colors[-1] <= best:
                return
            vertex = order.pop()
            colors.pop()
            bit = 1 << vertex
            expand(size + 1, clique_mask | bit, candidates & adjacency_masks[vertex])
            candidates &= ~bit
            if size + candidates.bit_count() <= best:
                return

    expand(0, 0, (1 << n) - 1)
    return best, _vertices_from_mask(best_mask)


def _complement_masks(graph: dict[int, set[int]], n: int) -> list[int]:
    if n < 1:
        raise ValueError("n must be positive")

    original_masks = [0] * n
    for vertex in range(1, n + 1):
        mask = 0
        for neighbor in graph.get(vertex, set()):
            if 1 <= neighbor <= n and neighbor != vertex:
                mask |= 1 << (neighbor - 1)
        original_masks[vertex - 1] = mask

    all_vertices = (1 << n) - 1
    complement_masks = []
    for index in range(n):
        self_bit = 1 << index
        complement_masks.append(all_vertices & ~self_bit & ~original_masks[index])
    return complement_masks


def find_max_independent_set(graph: dict[int, set[int]], n: int) -> tuple[int, list[int]]:
    return _maximum_clique(_complement_masks(graph, n), n)


def max_independent_set_size(graph: dict[int, set[int]], n: int) -> int:
    size, _vertices = find_max_independent_set(graph, n)
    return size


def solve_divisibility_subset_problem(problem: str) -> dict | None:
    detection = detect_divisibility_subset_problem(problem)
    if detection is None:
        return None

    n = int(detection["n"])
    graph = build_divisibility_graph(n)
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    isolated_vertices = [vertex for vertex, neighbors in graph.items() if not neighbors]
    alpha, independent_set = find_max_independent_set(graph, n)
    k_value = alpha + 1
    verification_note = "最小 K 等于最大独立集大小 alpha 加 1；工具通过精确最大独立集搜索得到 alpha。"
    solution = (
        f"构造图 G，顶点为 1 到 {n}。若 a+b 整除 ab，则在 a,b 之间连边。"
        f"不含这种二元组的子集正是 G 的独立集。本地精确搜索得到最大独立集大小为 {alpha}，"
        f"因此任意 K 元子集必含边的最小值为 alpha+1={k_value}。"
    )
    return {
        "tool_name": "divisibility_subset_graph",
        "final_answer": str(k_value),
        "details": {
            "n": n,
            "edge_count": edge_count,
            "isolated_vertices": isolated_vertices,
            "isolated_count": len(isolated_vertices),
            "max_independent_set_size": alpha,
            "one_max_independent_set": independent_set,
            "K": k_value,
            "verification_note": verification_note,
        },
        "solution": solution,
    }
