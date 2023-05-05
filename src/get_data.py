"""Data access object that wraps all the different resources regarding one datasource"""

from pathlib import Path
from .dataset import DocumentSet, QuerySet, RankingSet
from typing import List, Generator
from .faiss import FaissIndexWrapper
from .spacy import SpacyIndex
import pandas as pd

class Data:
    def __init__(self, data_path: Path):
        self.path = Path(data_path)

    def get_docset(self) -> DocumentSet: pass

    def get_queryset(self, queryset_name='queries') -> QuerySet:
        path = self.path / 'queries' / queryset_name
        if not path.exists():
            raise FileNotFoundError(f"No queryset '{queryset_name}' in {str(self.path / 'queries')}")
        return QuerySet(path)

    def list_qsets(self) -> List[str]:
        return [d.name for d in (self.path / 'queries').glob('*') if d.is_dir()]

    def get_rankings(self, queryset_name='queries', ranking_method='neural') -> RankingSet:
        rset_name = f"{ranking_method}_{queryset_name}.csv"
        path = self.path / 'rankings' / rset_name
        if not path.exists():
            raise FileNotFoundError(f"No ranking '{rset_name}' in {str(self.path / 'rankings')}")
        return RankingSet(df_path=path)
    
    def list_rankingsets(self) -> List[str]:
        return [d.name for d in (self.path / 'rankings').glob('*.csv')]

    def list_probe_qsets(self) -> List[str]:
        return [d.name for d in (self.path / 'probes').glob('*') if d.is_dir()]

    def list_probes(self, qset_name='queries') -> List[str]:
        return [d.name for d in (self.path / 'probes' / qset_name).glob('*.csv')]

    def has_faiss(self):
        return Path(self.path / 'indices/faiss').exists()

    def has_lucene(self):
        return Path(self.path / 'indices/lucene').exists()

    def get_lucene_index(self):
        assert self.has_lucene()
        from .lucene import LuceneIndex  # import this here because we cannot import lucene when running on colab
        return LuceneIndex(self.path / 'indices/lucene', 'en')

    def get_faiss_index(self):
        assert self.has_faiss()
        return FaissIndexWrapper(self.path / 'indices/faiss')

    def has_document_spacy_index(self):
        return Path(self.path / 'indices/spacy').exists()

    def get_document_spacy(self):
        assert self.has_document_spacy_index()
        return SpacyIndex(self.path / 'indices/spacy')
        


class CqaDupStackData(Data):
    def __init__(self, name):
        super().__init__(Path(f'data/cqadupstack/{name}'))

    def get_docset(self) -> DocumentSet:
        documents_df = pd.read_csv(self.path / 'documents.csv')
        return DocumentSet(df=documents_df, id_field_name='doc_id', default_text_field_name='value')

class CqaDupStackCollector:
    @staticmethod
    def iter_datasets() -> Generator[CqaDupStackData, None, None]:
        yield from [CqaDupStackData(path.stem) for path in sorted(Path('data/cqadupstack').glob('*'))]

    @staticmethod
    def list_datasets() -> List[str]:
        return [path.stem for path in sorted(Path('data/cqadupstack').glob('*'))]