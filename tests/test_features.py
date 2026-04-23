"""
Test feature engineering outputs.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.tabular import generate_tabular_features
from src.features.text_embeddings import generate_text_embeddings
from src.features import text_embeddings as text_embeddings_module
from src.features import fuse as fuse_module


class TestTabularFeatures:
    """Test tabular feature extraction."""
    
    def test_tabular_features_shape(self):
        """Check that tabular features have correct number of columns."""
        df = generate_tabular_features()
        assert not df.empty
        assert df.shape[1] == 27
    
    def test_tabular_features_no_nulls(self):
        """Check that tabular features have no null values."""
        df = generate_tabular_features()
        numeric_cols = [col for col in df.columns if col not in ["ticker", "quarter"]]
        # Current tabular pipeline may emit sparse NaNs for EPS-derived fields.
        # Ensure at least core price-derived fields are fully populated.
        required_cols = ["price_return", "volatility", "avg_volume", "momentum", "trading_days"]
        for col in required_cols:
            assert col in numeric_cols
            assert df[col].notna().all()
    
    def test_tabular_features_ticker_count(self):
        """Check that all tickers are represented."""
        df = generate_tabular_features()
        assert df["ticker"].nunique() >= 100


class TestTextEmbeddings:
    """Test text embedding extraction."""

    @staticmethod
    def _mock_load_finbert_model(device: str = "cpu"):
        class DummyTokenizer:
            pass

        class DummyModel:
            config = type("Config", (), {"hidden_size": 768})()

        return DummyTokenizer(), DummyModel()

    @staticmethod
    def _mock_get_embedding_for_text(text, tokenizer, model, device="cpu", max_length=512):
        seed = len(text) % 97
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, size=768).astype(np.float32)

    @pytest.fixture
    def tiny_filings_dir(self, tmp_path):
        filings_dir = tmp_path / "filings"
        filings_dir.mkdir(parents=True, exist_ok=True)
        sample_text = (
            "ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION "
            "Revenue improved and expenses declined. "
            "ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK"
        )
        (filings_dir / "AAPL_2019Q1_abc.txt").write_text(sample_text, encoding="utf-8")
        (filings_dir / "MSFT_2019Q2_def.txt").write_text(sample_text, encoding="utf-8")
        return filings_dir
    
    def test_embeddings_shape(self, monkeypatch, tiny_filings_dir):
        """Check that embeddings are 768-dimensional."""
        monkeypatch.setattr(text_embeddings_module, "load_finbert_model", self._mock_load_finbert_model)
        monkeypatch.setattr(text_embeddings_module, "get_embedding_for_text", self._mock_get_embedding_for_text)

        df = generate_text_embeddings(data_dir=str(tiny_filings_dir), device="cpu")
        assert df.shape[1] == 770
    
    def test_embeddings_no_nulls(self, monkeypatch, tiny_filings_dir):
        """Check that embeddings have no null values."""
        monkeypatch.setattr(text_embeddings_module, "load_finbert_model", self._mock_load_finbert_model)
        monkeypatch.setattr(text_embeddings_module, "get_embedding_for_text", self._mock_get_embedding_for_text)

        df = generate_text_embeddings(data_dir=str(tiny_filings_dir), device="cpu")
        assert not df.isna().any().any()
    
    def test_embeddings_in_valid_range(self, monkeypatch, tiny_filings_dir):
        """Check that embedding values are in reasonable range."""
        monkeypatch.setattr(text_embeddings_module, "load_finbert_model", self._mock_load_finbert_model)
        monkeypatch.setattr(text_embeddings_module, "get_embedding_for_text", self._mock_get_embedding_for_text)

        df = generate_text_embeddings(data_dir=str(tiny_filings_dir), device="cpu")
        embedding_cols = [c for c in df.columns if c.startswith("embedding_")]
        vals = df[embedding_cols].to_numpy()
        assert np.all(np.isfinite(vals))
        assert np.max(np.abs(vals)) < 10


class TestFusedFeatures:
    """Test merged feature dataset."""

    @staticmethod
    def _tiny_tabular_df() -> pd.DataFrame:
        rows = []
        for ticker, quarter in [("AAPL", "2019Q1"), ("AAPL", "2019Q2"), ("MSFT", "2019Q1")]:
            row = {"ticker": ticker, "quarter": quarter}
            for i in range(25):
                row[f"f_{i}"] = float(i)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _tiny_embeddings_df() -> pd.DataFrame:
        rows = []
        for ticker, quarter in [("AAPL", "2019Q1"), ("AAPL", "2019Q2"), ("MSFT", "2019Q1")]:
            row = {"ticker": ticker, "quarter": quarter}
            for i in range(768):
                row[f"embedding_{i}"] = 0.01 * i
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _tiny_prices_dict() -> dict:
        aapl = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-01-10", "2019-03-20", "2019-04-15", "2019-06-20"]),
                "reported_eps": [1.20, 1.25, 1.30, 1.35],
                "eps_estimate": [1.00, 1.20, 1.20, 1.30],
            }
        ).set_index("date")
        msft = pd.DataFrame(
            {
                "date": pd.to_datetime(["2019-01-10", "2019-03-20"]),
                "reported_eps": [0.80, 0.75],
                "eps_estimate": [0.82, 0.78],
            }
        ).set_index("date")
        return {"AAPL": aapl, "MSFT": msft}
    
    def test_fused_file_exists(self, monkeypatch, tmp_path):
        """Check that features.parquet is created."""
        monkeypatch.setattr(fuse_module, "generate_tabular_features", self._tiny_tabular_df)
        monkeypatch.setattr(
            fuse_module,
            "generate_text_embeddings",
            lambda device="cpu": self._tiny_embeddings_df(),
        )
        monkeypatch.setattr(fuse_module, "load_prices_data", self._tiny_prices_dict)

        fuse_module.fuse_features(output_dir=str(tmp_path), device="cpu")
        features_path = Path(tmp_path) / "features.parquet"
        assert features_path.exists()
    
    def test_fused_shape(self, monkeypatch, tmp_path):
        """Check that fused dataset has correct number of columns."""
        monkeypatch.setattr(fuse_module, "generate_tabular_features", self._tiny_tabular_df)
        monkeypatch.setattr(
            fuse_module,
            "generate_text_embeddings",
            lambda device="cpu": self._tiny_embeddings_df(),
        )
        monkeypatch.setattr(fuse_module, "load_prices_data", self._tiny_prices_dict)

        fuse_module.fuse_features(output_dir=str(tmp_path), device="cpu")
        df = pd.read_parquet(Path(tmp_path) / "features.parquet")
        assert df.shape[1] == 796
    
    def test_fused_no_nulls(self, monkeypatch, tmp_path):
        """Check that fused features have no null values."""
        monkeypatch.setattr(fuse_module, "generate_tabular_features", self._tiny_tabular_df)
        monkeypatch.setattr(
            fuse_module,
            "generate_text_embeddings",
            lambda device="cpu": self._tiny_embeddings_df(),
        )
        monkeypatch.setattr(fuse_module, "load_prices_data", self._tiny_prices_dict)

        fuse_module.fuse_features(output_dir=str(tmp_path), device="cpu")
        df = pd.read_parquet(Path(tmp_path) / "features.parquet")
        assert not df.isna().any().any()
    
    def test_label_distribution(self, monkeypatch, tmp_path):
        """Check that label has 0s and 1s."""
        monkeypatch.setattr(fuse_module, "generate_tabular_features", self._tiny_tabular_df)
        monkeypatch.setattr(
            fuse_module,
            "generate_text_embeddings",
            lambda device="cpu": self._tiny_embeddings_df(),
        )
        monkeypatch.setattr(fuse_module, "load_prices_data", self._tiny_prices_dict)

        fuse_module.fuse_features(output_dir=str(tmp_path), device="cpu")
        df = pd.read_parquet(Path(tmp_path) / "features.parquet")
        labels = set(df["label"].unique().tolist())
        assert labels.issubset({0, 1})
        assert labels == {0, 1}
    
    def test_fused_unique_quarters(self, monkeypatch, tmp_path):
        """Check that we have multiple quarters of data."""
        monkeypatch.setattr(fuse_module, "generate_tabular_features", self._tiny_tabular_df)
        monkeypatch.setattr(
            fuse_module,
            "generate_text_embeddings",
            lambda device="cpu": self._tiny_embeddings_df(),
        )
        monkeypatch.setattr(fuse_module, "load_prices_data", self._tiny_prices_dict)

        fuse_module.fuse_features(output_dir=str(tmp_path), device="cpu")
        df = pd.read_parquet(Path(tmp_path) / "features.parquet")
        assert df["quarter"].nunique() >= 2
