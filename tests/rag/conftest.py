import pandas as pd
import pytest
from rag_helpers import HashEmbedder, make_docs


@pytest.fixture
def docs() -> pd.DataFrame:
    return make_docs()


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()
