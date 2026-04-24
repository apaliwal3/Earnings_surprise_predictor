"""
Extract 768-dimensional FinBERT embeddings from 10-Q MD&A sections.

Input: data/raw/filings/*.txt (10-Q documents)
Output: DataFrame with columns [ticker, quarter, embedding_0, ..., embedding_767]
"""
import os
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from src.data.download_prices import load_sp500_tickers as local_sp500_tickers


def resolve_device(device: str = "mps") -> str:
    """Resolve device preference, falling back to CPU when MPS is unavailable."""
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return device


def load_local_sp500_tickers() -> list:
    """
    Load S&P 500 ticker list (fallback to hardcoded list if needed).
    """
    try:
        return list(local_sp500_tickers())
    except Exception:  # noqa: BLE001
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def extract_mda_text(filing_text: str) -> str:
    """
    Extract the MD&A (Management's Discussion and Analysis) section from 10-Q text.
    
    MD&A typically starts after "Item 2" and ends before "Item 3" or similar.
    
    Args:
        filing_text: Raw 10-Q text from SEC EDGAR
    
    Returns:
        str: Extracted MD&A text (cleaned)
    """
    if not filing_text:
        return ""

    patterns = [
        r"item\s*2\.?\s*management['’]s\s+discussion\s+and\s+analysis.*?(?=item\s*3\.?)",
        r"item\s*2\.?\s*management['’]s\s+discussion.*?(?=item\s*3\.?)",
        r"item\s*2\.?\s*.*?(?=item\s*3\.?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, filing_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(0)
            text = re.sub(r"\s+", " ", text).strip()
            return text

    # Fallback: return leading chunk so embedding generation can proceed.
    return re.sub(r"\s+", " ", filing_text[:8000]).strip()


def load_finbert_model(device: str = "mps"):
    """
    Load FinBERT tokenizer and model.
    
    Args:
        device: "cpu" or "mps" (for Apple Silicon)
    
    Returns:
        (tokenizer, model)
    """
    device = resolve_device(device)

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    model.to(device)
    model.eval()
    return tokenizer, model


def get_embedding_for_text(text: str, tokenizer, model, device: str = "mps", max_length: int = 512) -> np.ndarray:
    """
    Given text, tokenize, get [CLS] token embedding (768-dim).
    
    Args:
        text: Text to embed
        tokenizer: FinBERT tokenizer
        model: FinBERT model
        device: "cpu" or "mps"
        max_length: Max token length for truncation
    
    Returns:
        np.ndarray: 768-dimensional embedding
    """
    device = resolve_device(device)
    hidden_size = getattr(model.config, "hidden_size", 768)
    if not text:
        return np.zeros(hidden_size, dtype=np.float32)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)


def extract_quarter_from_filename(filename: str) -> str:
    """
    Extract quarter from filename like "AAPL_10-Q_2015-12-31.txt" -> "2015Q4"
    
    Args:
        filename: e.g. "AAPL_10-Q_2015-12-31.txt"
    
    Returns:
        str: quarter in format "YYYYQX"
    """
    direct_q = re.search(r"(\d{4}Q[1-4])", filename)
    if direct_q:
        return direct_q.group(1)

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if not date_match:
        return "unknownQ"

    filing_date = pd.to_datetime(date_match.group(1), errors="coerce")
    if pd.isna(filing_date):
        return "unknownQ"

    quarter = ((int(filing_date.month) - 1) // 3) + 1
    return f"{int(filing_date.year)}Q{quarter}"


def generate_text_embeddings(data_dir: str = "data/raw/filings", device: str = "mps") -> pd.DataFrame:
    """
    Process all 10-Q filings, extract MD&A, generate embeddings.
    
    Args:
        data_dir: Path to filings directory
        device: "cpu" or "mps"
    
    Returns:
        pd.DataFrame: columns [ticker, quarter, embedding_0, ..., embedding_767]
    """
    device = resolve_device(device)
    filings_path = Path(data_dir)
    if not filings_path.exists():
        return pd.DataFrame(columns=["ticker", "quarter"])

    tokenizer, model = load_finbert_model(device=device)

    rows = []
    txt_files = sorted(filings_path.glob("*.txt"))

    for idx, file_path in enumerate(txt_files, start=1):
        filename = file_path.name
        ticker = filename.split("_")[0].upper()
        quarter = extract_quarter_from_filename(filename)

        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        mda_text = extract_mda_text(raw_text)
        embedding = get_embedding_for_text(
            mda_text,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        row = {"ticker": ticker, "quarter": quarter}
        row.update({f"embedding_{i}": float(val) for i, val in enumerate(embedding)})
        rows.append(row)

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(txt_files)} filings")

    if not rows:
        return pd.DataFrame(columns=["ticker", "quarter"])

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "quarter"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    embeddings_df = generate_text_embeddings(device="mps")  # or "mps" for Apple Silicon
    print(f"Generated embeddings: {embeddings_df.shape}")
    print(embeddings_df.head())
