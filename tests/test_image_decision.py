from encoders.image_encoder import decide_harmful

FLAGS = [True, True, False, False]  # two harmful prompts, two benign


def test_unrelated_image_forced_onto_harmful_prompt_is_rejected():
    # Skunk-like case: softmax piles onto a threat prompt (0.65) but the raw
    # cosine similarity is low -> must NOT be flagged.
    assert decide_harmful([0.65, 0.1, 0.15, 0.1], [0.17, 0.15, 0.16, 0.15], FLAGS, 0.25, 0.22, 0.15) == [False] * 4


def test_genuine_threat_is_flagged():
    out = decide_harmful([0.7, 0.05, 0.15, 0.1], [0.29, 0.2, 0.22, 0.2], FLAGS, 0.25, 0.22, 0.15)
    assert out == [True, False, False, False]


def test_kitchen_knife_benign_prompt_wins_margin():
    # Harmful prompt only slightly ahead of the benign cooking prompt -> rejected.
    assert decide_harmful([0.42, 0.05, 0.40, 0.13], [0.26, 0.2, 0.26, 0.2], FLAGS, 0.25, 0.22, 0.15)[0] is False
