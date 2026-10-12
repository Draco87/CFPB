"""Text cleaning and label normalisation shared by all modules."""
import re

# CFPB renamed products over the years; collapse old and new names into one label.
# After running scripts/00_combine_archive.py, check the printed product counts
# and add anything large that is missing.
PRODUCT_MAP = {
    "Credit reporting": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Debt or credit management": "Debt collection",
    "Credit card": "Credit / prepaid card",
    "Prepaid card": "Credit / prepaid card",
    "Credit card or prepaid card": "Credit / prepaid card",
    "Bank account or service": "Bank account",
    "Checking or savings account": "Bank account",
    "Mortgage": "Mortgage",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan",
    "Consumer Loan": "Vehicle loan",
    "Payday loan": "Payday / personal loan",
    "Payday loan, title loan, or personal loan": "Payday / personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday / personal loan",
    "Money transfers": "Money transfer / virtual currency",
    "Virtual currency": "Money transfer / virtual currency",
    "Money transfer, virtual currency, or money service": "Money transfer / virtual currency",
}

_MASK_RE = re.compile(r"\b[X]{2,}(?:/[X]{2,})*\b|\{\$[\d.,]+\}")  # XXXX, XX/XX/XXXX, {$1200.00}
_SPACE_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Remove CFPB redaction masks and collapse whitespace."""
    s = _MASK_RE.sub(" ", s)
    return _SPACE_RE.sub(" ", s).strip()
