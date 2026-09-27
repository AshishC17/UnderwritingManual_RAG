from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.graph.pipeline import (
    _build_rerank_assignments,
    _pool_candidates,
    _select_rerank_candidates,
)


def _hit(chunk_id: str, token_count: int = 10, score: float = 0.5):
    return SimpleNamespace(
        payload={
            "chunk_id": chunk_id,
            "text": f"text for {chunk_id}",
            "token_count": token_count,
        },
        score=score,
    )


class CandidateSelectionTests(unittest.TestCase):
    def test_duplicate_is_owned_by_its_best_rrf_rank(self):
        query_one = [_hit(f"q1-{i}") for i in range(1, 6)] + [_hit("flow")]
        query_two = [_hit(f"q2-{i}") for i in range(1, 5)] + [_hit("flow")]

        pool = _pool_candidates(["disposition", "route"], [query_one, query_two])

        self.assertEqual(2, len(pool["flow"]["matches"]))
        self.assertEqual("route", pool["flow"]["owner_sub_query"])
        self.assertEqual(1, pool["flow"]["owner_sub_query_index"])
        self.assertEqual(5, pool["flow"]["owner_rrf_rank"])

    def test_candidate_cap_is_global_and_interleaves_subqueries(self):
        hit_lists = [
            [_hit(f"q{query}-{rank}") for rank in range(1, 5)]
            for query in range(3)
        ]
        pool = _pool_candidates(["q0", "q1", "q2"], hit_lists)

        selected = _select_rerank_candidates(
            pool,
            max_candidates=6,
            max_tokens=10_000,
        )

        self.assertEqual(6, len(selected))
        self.assertEqual(
            [0, 1, 2, 0, 1, 2],
            [record["owner_sub_query_index"] for record in selected.values()],
        )
        self.assertEqual(
            [1, 1, 1, 2, 2, 2],
            [record["owner_rrf_rank"] for record in selected.values()],
        )

    def test_token_cap_is_shared_across_all_subqueries(self):
        pool = _pool_candidates(
            ["q0", "q1", "q2"],
            [[_hit("a", 100)], [_hit("b", 100)], [_hit("c", 100)]],
        )

        selected = _select_rerank_candidates(
            pool,
            max_candidates=24,
            max_tokens=250,
        )

        self.assertEqual(["a", "b"], list(selected))
        self.assertEqual(200, sum(
            record["chunk"]["token_count"] for record in selected.values()
        ))

    def test_tied_best_match_uses_only_spare_pair_budget(self):
        pool = _pool_candidates(
            ["q0", "q1"],
            [[_hit("shared"), _hit("a")], [_hit("shared"), _hit("b")]],
        )
        selected = _select_rerank_candidates(pool, max_candidates=4, max_tokens=100)

        assignments = _build_rerank_assignments(
            selected,
            max_pairs=4,
            max_tokens=100,
        )

        shared = [a for a in assignments if a["chunk"]["chunk_id"] == "shared"]
        self.assertEqual(4, len(assignments))
        self.assertEqual(2, len(shared))
        self.assertEqual({0, 1}, {a["sub_query_index"] for a in shared})
        self.assertEqual(1, sum(a["is_primary"] for a in shared))

        capped = _build_rerank_assignments(
            selected,
            max_pairs=3,
            max_tokens=100,
        )
        self.assertEqual(3, len(capped))
        self.assertEqual(
            1,
            sum(a["chunk"]["chunk_id"] == "shared" for a in capped),
        )


if __name__ == "__main__":
    unittest.main()
