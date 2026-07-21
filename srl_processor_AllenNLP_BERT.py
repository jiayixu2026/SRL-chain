import sys
import string
import spacy

_srl_predictor = None
_srl_load_failed = False
_nlp_spacy = None


def get_spacy_nlp():
    global _nlp_spacy
    if _nlp_spacy is None:
        try:
            _nlp_spacy = spacy.load("en_core_web_sm")
        except Exception:
            print("Warning: en_core_web_sm not found, verb-free sentences will use simple heuristics.")
            print("Try running: python -m spacy download en_core_web_sm")
            _nlp_spacy = None
    return _nlp_spacy


def get_srl_predictor():
    global _srl_predictor, _srl_load_failed

    if _srl_load_failed:
        return None

    if _srl_predictor is None:
        try:
            from allennlp.models.archival import load_archive
            from allennlp_models.structured_prediction.predictors.srl import SemanticRoleLabelerPredictor

            print("Loading SRL model (first run may take 1-2 minutes)...")
            archive = load_archive(
                'https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz',
                cuda_device=-1
            )
            _srl_predictor = SemanticRoleLabelerPredictor(archive.model, archive.dataset_reader)
            print("SRL model loaded successfully!")
        except ImportError as e:
            print(f"Missing dependencies: {e}")
            print("Please install: pip install allennlp allennlp-models")
            _srl_predictor = None
            _srl_load_failed = True
        except Exception as e:
            print(f"SRL model loading failed: {e}")
            _srl_predictor = None
            _srl_load_failed = True
    return _srl_predictor


def get_role_interpretation(tag: str, linked_to_aspect: bool = False) -> str:
    base = ""
    if tag.startswith('B-V') or tag.startswith('I-V'):
        base = "Predicate (Action)"
    elif tag.startswith('B-ARG0') or tag.startswith('I-ARG0'):
        base = "Who is acting"
    elif tag.startswith('B-ARG1') or tag.startswith('I-ARG1'):
        base = "What is evaluated"
    elif tag.startswith('B-ARG2') or tag.startswith('I-ARG2'):
        base = "How it is evaluated"
    elif 'ARGM' in tag:
        base = "Modifier"
    elif tag == 'O':
        return "Not important"
    else:
        return "-"

    if linked_to_aspect:
        base += " (linked to aspect)"
    return base


def find_aspect_indices(words, aspect):
    if not aspect:
        return set()
    aspect_tokens = aspect.lower().split()
    if not aspect_tokens:
        return set()
    for i in range(len(words) - len(aspect_tokens) + 1):
        if all(words[i + j].lower() == aspect_tokens[j] for j in range(len(aspect_tokens))):
            return set(range(i, i + len(aspect_tokens)))
    return set()


def srl_to_conll_style_v2(sentence: str, srl_result: dict, aspect: str = None) -> str:
    words = srl_result.get('words', [])
    if not words:
        words = [w.strip(string.punctuation) for w in sentence.split()]
        words = [w for w in words if w]

    verbs = srl_result.get('verbs', [])
    aspect_indices = find_aspect_indices(words, aspect)

    if not verbs:
        lines = []
        lines.append("=" * 80)
        lines.append("SEMANTIC ROLE LABELING (SRL) - No predicate detected")
        lines.append("=" * 80)
        lines.append("NOTE: No clear verb predicate found. Using heuristic based on aspect and POS.")
        lines.append("")
        lines.append(f"{'ID':<4} {'WORD':<18} {'ROLE':<15} {'INTERPRETATION':<25} {'ASPECT':<8} {'NEG':<4} {'MOD':<4}")
        lines.append("-" * 80)

        nlp = get_spacy_nlp()
        pos_tags = []
        if nlp is not None:
            doc = nlp(sentence)
            spacy_tokens = [token.text for token in doc]
            if len(spacy_tokens) == len(words):
                pos_tags = [token.pos_ for token in doc]
            else:
                pos_tags = [''] * len(words)
        else:
            pos_tags = [''] * len(words)

        modal_set = {'can', 'could', 'may', 'might', 'must', 'should', 'would', 'seem', 'appear', 'seems', 'appears'}

        for idx, word in enumerate(words, 1):
            idx0 = idx - 1
            is_aspect = "← aspect" if idx0 in aspect_indices else ""
            neg = "✓" if word.lower() in {'not', "n't", 'never', 'no'} else ""
            mod = "✓" if word.lower() in modal_set else ""

            if idx0 in aspect_indices:
                role = "ASPECT-TARGET"
                interpretation = "Evaluation target"
            else:
                pos = pos_tags[idx0] if idx0 < len(pos_tags) else ''
                if pos == 'ADJ' or pos == 'ADV':
                    role = "ASPECT-DESCRIPTOR"
                    interpretation = "Evaluation descriptor"
                else:
                    role = "OTHER"
                    interpretation = "Other word"

            lines.append(f"{idx:<4} {word:<18} {role:<15} {interpretation:<25} {is_aspect:<8} {neg:<4} {mod:<4}")

        lines.append("=" * 80)
        lines.append("")
        lines.append("SRL ROLE KEY:")
        lines.append("  ASPECT-TARGET   = What is being evaluated (based on aspect phrase)")
        lines.append("  ASPECT-DESCRIPTOR = Word describing the evaluation (adjective/adverb)")
        lines.append("  OTHER           = Other words")
        lines.append("  NEG = Negation marker")
        lines.append("  MOD = Modal/hedging word")
        return "\n".join(lines)

    merged_tags = ['O'] * len(words)
    negated_flags = [''] * len(words)
    modal_flags = [''] * len(words)

    for verb_info in verbs:
        tags = verb_info.get('tags', [])
        for i, tag in enumerate(tags):
            if i < len(merged_tags) and tag != 'O' and merged_tags[i] == 'O':
                merged_tags[i] = tag

        if verb_info.get('negated', False):
            for i, tag in enumerate(tags):
                if tag.startswith('B-V') and i < len(negated_flags):
                    negated_flags[i] = '✓'
                    break

    modal_set = {'can', 'could', 'may', 'might', 'must', 'should', 'would', 'seem', 'appear', 'seems', 'appears'}
    for i, word in enumerate(words):
        if word.lower() in modal_set:
            modal_flags[i] = '✓'

    for i, word in enumerate(words):
        if word.lower() in {'not', "n't", 'never', 'no'}:
            negated_flags[i] = '✓'

    main_verb = verbs[0].get('verb', '') if verbs else ''

    aspect_role_tag = None
    for idx in aspect_indices:
        if idx < len(merged_tags) and merged_tags[idx] != 'O':
            aspect_role_tag = merged_tags[idx]
            break

    lines = []
    lines.append("=" * 80)
    lines.append("SEMANTIC ROLE LABELING (SRL)")
    lines.append("=" * 80)
    lines.append(f"Predicate (Main Verb): {main_verb}")
    lines.append("")

    if aspect and aspect_indices:
        sorted_idx = sorted(aspect_indices)
        start_id = sorted_idx[0] + 1
        end_id = sorted_idx[-1] + 1
        role_str = get_role_interpretation(aspect_role_tag) if aspect_role_tag else "unknown role"
        note = f"Note: The aspect phrase '{aspect}' spans words {start_id}-{end_id} and functions as {role_str}."
        lines.append(note)
        lines.append("")

    lines.append(f"{'ID':<4} {'WORD':<18} {'SRL-ROLE':<15} {'INTERPRETATION':<30} {'ASPECT':<8} {'NEG':<4} {'MOD':<4}")
    lines.append("-" * 80)

    for idx, (word, tag) in enumerate(zip(words, merged_tags), 1):
        idx0 = idx - 1
        is_aspect = "← aspect" if idx0 in aspect_indices else ""

        linked = False
        if aspect_role_tag and tag != 'O' and (idx0 not in aspect_indices):
            if 'ARG' in tag or 'ARGM' in tag:
                linked = True

        interpretation = get_role_interpretation(tag, linked_to_aspect=linked)
        neg = negated_flags[idx0]
        mod = modal_flags[idx0]
        lines.append(f"{idx:<4} {word:<18} {tag:<15} {interpretation:<30} {is_aspect:<8} {neg:<4} {mod:<4}")

    lines.append("=" * 80)
    lines.append("")
    lines.append("SRL ROLE KEY:")
    lines.append("  V       = Verb/Predicate (the action or state being described)")
    lines.append("  ARG0    = Agent (Who is performing the action or doing the evaluation)")
    lines.append("  ARG1    = Patient/Theme (What is being acted upon, or what is being evaluated)")
    lines.append("  ARG2    = Attribute/Result (How it is being evaluated, or the quality/description)")
    lines.append("  ARGM-*  = Modifiers (Time, Location, Manner, etc.)")
    lines.append("  O       = Not important / Irrelevant word")
    lines.append("")
    lines.append("Additional columns:")
    lines.append("  ASPECT  = '← aspect' if the word is part of the target aspect phrase")
    lines.append("  NEG     = '✓' if the word is a negation marker or the predicate is negated")
    lines.append("  MOD     = '✓' if the word is a modal verb or hedging expression")
    lines.append("  '(linked to aspect)' appears for arguments sharing the predicate with the aspect.")

    return "\n".join(lines)


def get_semantic_role_labeling_v2(sentence, aspect=None):
    predictor = get_srl_predictor()
    if predictor is None:
        return "SRL predictor not available. Please install allennlp and allennlp-models."

    try:
        prediction = predictor.predict(sentence=sentence)
        return srl_to_conll_style_v2(sentence, prediction, aspect)
    except Exception as e:
        return f"Error in SRL: {str(e)}"


def test_srl():
    test_sentences = [
        ("The bread is top notch as well.", "bread"),
        ("I have to say they have one of the fastest delivery times in the city.", "delivery times"),
        ("BEST spicy tuna roll, great asian salad.", "spicy tuna roll"),
        ("Food is always fresh and hot- ready to eat!", "Food"),
        ("Great food!", "food"),
        ("Absolutely delicious.", "Absolutely delicious"),
        ("The service was not good.", "service"),
        ("The price might be reasonable.", "price"),
    ]

    print("=" * 80)
    print("SRL v2 Test (with NEG/MOD columns, linkage markers, phrase notes)")
    print("=" * 80)

    for sent, aspect in test_sentences:
        print("\n" + "=" * 80)
        print(f"Sentence: {sent}")
        print(f"Aspect: {aspect}")
        result = get_semantic_role_labeling_v2(sent, aspect=aspect)
        print(result)


if __name__ == '__main__':
    test_srl()