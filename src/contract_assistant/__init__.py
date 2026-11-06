"""Private, evaluated contract assistant.

Measures how much planted personal data leaves the machine, lands in the search index, and appears in answers of a
retrieval-augmented contract assistant, and whether its answers are grounded in the passages they cite.
"""

from contract_assistant.config import Config
from contract_assistant.leakage import leaks
from contract_assistant.redaction import Redacted, redact
from contract_assistant.synthetic import inject, make_person

__all__ = ["Config", "Redacted", "inject", "leaks", "make_person", "redact"]
__version__ = "0.1.0"
