"""
preprocessing.py — Text preprocessing pipeline.

The original notebook passes raw text strings DIRECTLY into the TF Hub layer:

    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
                   input_shape=[], dtype=tf.string, trainable=True)

The Swivel embedding layer performs its own internal tokenization and
lookup — no external tokenizer, no manual lowercasing, no stopword removal.
The notebook applies ZERO preprocessing before the Hub layer.

Therefore this module does ONLY:
  1. Strip leading/trailing whitespace (safe for any text model)
  2. Validate non-empty after stripping

Nothing else.  Do NOT add: lowercasing, HTML stripping, punctuation removal,
stopwords, stemming.  The model was trained on raw strings.
"""

from app.config import settings


def preprocess(text: str) -> str:
    """
    Prepare a raw input string for inference.

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    str
        Cleaned text ready to be passed to the model.

    Raises
    ------
    ValueError
        If the text is empty or whitespace-only after stripping.
    """
    cleaned = text.strip()

    if not cleaned:
        raise ValueError("Text cannot be empty or whitespace-only.")

    if len(cleaned) > settings.MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text exceeds maximum length of {settings.MAX_TEXT_LENGTH} characters. "
            f"Received {len(cleaned)} characters."
        )

    return cleaned
