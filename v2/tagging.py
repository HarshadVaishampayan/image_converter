"""Zero-shot image tagging and text relevance on SigLIP's calibrated scale.

SigLIP is trained with a sigmoid loss, so `logit = scale * cos(text, image) + bias`
is comparable across images and queries (cosine distance alone is not). We use it
for two things:

* tagging: score each image against the TAXONOMY labels; keep the best ones and
  propagate them up the hierarchy (cherry blossom -> flower -> plant);
* searching: turn the query's own image-text logit, the tags and the description
  into one 0..1 relevance per image.
"""
import re
import threading

import numpy as np

from taxonomy import TAXONOMY, ancestors

# Logit -> confidence mapping: sigmoid((logit - CENTER) / WIDTH).
# Calibrated on SigLIP 2 so400m: true specific labels score about -6..0,
# unrelated ones -12 and below.
CENTER = -8.0
WIDTH = 1.5
MIN_TAG_LOGIT = -10.0     # tags below this are not stored
MAX_TAGS = 12
REL_GAP = 4.5             # ignore tags this far below the image's best tag (noise)

_lock = threading.Lock()
_vocab_cache = {}  # id(model) -> (labels, matrix)


def logit_params(model):
    """(scale, bias) for a SigLIPWrapper; None for models without a sigmoid head."""
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "logit_bias"):
        return None
    return float(inner.logit_scale.exp()), float(inner.logit_bias)


def supports(model):
    return logit_params(model) is not None


def _encode_texts(model, texts):
    emb = model.encode(texts, convert_to_tensor=True)
    arr = emb.float().cpu().numpy()
    return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)


def _vocab(model):
    with _lock:
        key = id(model)
        if key not in _vocab_cache:
            labels = list(TAXONOMY)
            mat = _encode_texts(model, [f"a photo of {label}" for label in labels])
            _vocab_cache[key] = (labels, mat)
        return _vocab_cache[key]


def confidence(logit):
    return 1.0 / (1.0 + np.exp(-(np.asarray(logit) - CENTER) / WIDTH))


def tag_embeddings(model, embeddings):
    """For each image embedding, a list of [label, logit] (best first)."""
    params = logit_params(model)
    if params is None:
        return [[] for _ in embeddings]
    scale, bias = params
    labels, mat = _vocab(model)
    E = np.asarray(embeddings, dtype=np.float32)
    E = E / np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-12)
    logits = scale * (E @ mat.T) + bias
    out = []
    for row in logits:
        idx = np.argsort(-row)[:MAX_TAGS]
        out.append([[labels[i], round(float(row[i]), 2)] for i in idx if row[i] >= MIN_TAG_LOGIT])
    return out


def confident_tags(tags):
    """Tags close enough to the image's best tag to be trusted."""
    if not tags:
        return []
    best = max(logit for _label, logit in tags)
    return [[label, logit] for label, logit in tags if logit >= best - REL_GAP]


def expand_tags(tags):
    """{term: confidence} for tags, their aliases and all their ancestors.

    A parent inherits the confidence of its most confident descendant.
    """
    terms = {}
    tags = confident_tags(tags)

    def put(text, conf):
        # each word of a label counts ("cherry", "blossom"), and so does the whole phrase
        words = [lemma(w) for w in text.split() if w not in _STOP]
        for key in {*words, " ".join(words)}:
            if conf > terms.get(key, 0.0):
                terms[key] = conf

    for label, logit in tags:
        conf = float(confidence(logit))
        for name in (label, *ancestors(label)):
            put(name, conf)
            for alias in TAXONOMY.get(name, ((), []))[1]:
                put(alias, conf)
    return terms


# ------------------------------------------------------------------ text ----
_STOP = {"a", "an", "the", "of", "with", "in", "on", "at", "and", "or", "to", "for", "by",
         "photo", "photos", "picture", "pictures", "image", "images", "pic", "pics",
         "some", "my", "me", "show", "find", "that", "this", "is", "are"}
_IRREGULAR = {"leaves": "leaf", "people": "person", "children": "child", "men": "man",
              "women": "woman", "mice": "mouse", "geese": "goose", "teeth": "tooth",
              "feet": "foot", "knives": "knife", "wolves": "wolf", "loaves": "loaf",
              "cherries": "cherry", "puppies": "puppy", "berries": "berry"}
_WORD = re.compile(r"[a-z0-9]+")


def lemma(word):
    """Very small English singularizer (flowers -> flower, blossoms -> blossom)."""
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def query_terms(text):
    return [lemma(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


def text_words(text):
    return {lemma(w) for w in _WORD.findall((text or "").lower())}


# --------------------------------------------------------------- ranking ----
DESCRIPTION_WEIGHT = 0.8   # a word in the description is good, not conclusive


def relevance(query, query_logits, tags_per_image, descriptions):
    """0..1 relevance per image for a text query.

    query_logits   : SigLIP logit of "a photo of {query}" vs each image
    tags_per_image : [[label, logit], ...] per image
    descriptions   : description text per image
    """
    terms = query_terms(query)
    phrase = " ".join(terms)
    visual = confidence(query_logits)
    scores = []
    for v, tags, desc in zip(visual, tags_per_image, descriptions):
        tag_terms = expand_tags(tags)
        words = text_words(desc)
        if terms:
            # every query term needs evidence from a tag or the description
            per_term = [max(tag_terms.get(t, 0.0), DESCRIPTION_WEIGHT if t in words else 0.0)
                        for t in terms]
            # softened AND: missing terms pull the score down hard
            text_score = float(np.mean(per_term)) * (0.5 + 0.5 * min(per_term))
            if phrase in tag_terms:  # whole phrase is a known label (e.g. "cherry blossom")
                text_score = max(text_score, tag_terms[phrase])
        else:
            text_score = 0.0
        # best single source, plus a small visual tie-breaker
        scores.append(min(1.0, max(float(v), text_score) + 0.05 * float(v)))
    return scores
