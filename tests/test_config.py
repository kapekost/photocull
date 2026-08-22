from photocull.config import ClusterConfig


def test_defaults_are_precision_first():
    c = ClusterConfig()
    # Measured against the real library during Phase 1a planning: true near-dups
    # (a real burst group) landed 0.21-0.42 apart, unrelated photos 0.83-1.26.
    # 0.40 sits just above the near-dup band with a wide empty gap after it --
    # see DECISIONS.md cluster-precision-over-recall.
    assert c.similarity_threshold == 0.40
    assert c.gap_seconds == 90.0
    assert c.ambiguity_margin == 0.05


def test_every_threshold_is_overridable():
    c = ClusterConfig(similarity_threshold=0.6, gap_seconds=30.0, ambiguity_margin=0.1)
    assert (c.similarity_threshold, c.gap_seconds, c.ambiguity_margin) == (0.6, 30.0, 0.1)


def test_score_weights_sum_to_one():
    c = ClusterConfig()
    assert abs(sum(c.weights.values()) - 1.0) < 1e-9


def test_sharpness_size_tolerance_defaults_to_exact_match():
    assert ClusterConfig().sharpness_size_tolerance == 1.0
