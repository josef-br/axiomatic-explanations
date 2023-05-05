"""
Data classes: DocumentSet, QuerySet, RankingSet, Ranking
"""
import pandas as pd
from typing import Generator, List, Tuple, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from .spacy import SpacyIndex
from itertools import starmap

class DocumentSet:
    """Entire document set is just a pd dataframe in-memory which we filter on"""
    def __init__(self, 
                df: pd.DataFrame = None,
                df_path = None,
                id_field_name: str = 'id',
                default_text_field_name='text',
                **kwargs):
        if (df is None and df_path is None) or \
            (df is not None and df_path is not None):
            raise ValueError("Must specify either df or df_path")

        if df is not None:
            self.dataframe = df
        elif df_path is not None:
            self.dataframe = pd.read_csv(df_path, **kwargs) 

        if id_field_name not in self.dataframe:
            raise ValueError(f"Specified id_field \"{id_field_name}\" not in dataframe")

        self.dataframe = self.dataframe.drop_duplicates(subset=id_field_name, keep='first') \
                                        .set_index(id_field_name)

        self.id_field_name = id_field_name
        self.default_text_field_name = default_text_field_name

    def get_all_document_ids(self) -> pd.Series:
        return self.dataframe.index.to_series()

    def get_document_batch_by_ids(self, ids: list, fields: List[str]=None) -> pd.DataFrame:
        if fields is None:
            fields = [self.id_field_name, self.default_text_field_name]
        elif self.id_field_name not in fields:
            fields.insert(0, self.id_field_name)
        return self.dataframe \
                .loc[ids] \
                .reset_index()[fields]

    def get_document_texts(self, doc_ids, field: str=None) -> str:
        """Get the text content of a single document"""
        return self.dataframe.loc[doc_ids][field if field is not None else self.default_text_field_name].values

    def get_document_text(self, doc_id, field: str=None) -> str:
        return str(self.get_document_texts([doc_id], field)[0])

    def num_documents(self) -> int:
        return self.dataframe.shape[0]

    def iter_for_indexing(self) -> Generator[Tuple[Union[str, int], str], None, None]:
        return self.dataframe[self.default_text_field_name].items()
    

class QuerySet:
    """Query sets are stored a bit more complicated. Coming from a pandas dataframe, a directory is created storing 
    the query texts, their embeddings according to the neural model, and their spacy processed data"""
    QUERIES_DF_FILENAME = 'queries.csv'
    QUERIES_EMBEDDING_FILENAME = 'embeddings.npy'
    SPACY_DIR = 'spacy'

    def __init__(self, data_path):
        data_path = Path(data_path)
        if not (data_path.is_dir() 
            and (data_path / self.QUERIES_DF_FILENAME).exists() 
            and (data_path / self.QUERIES_EMBEDDING_FILENAME).exists()):
            raise FileNotFoundError(f"data_path needs to be a directory containing {self.QUERIES_DF_FILENAME} and {self.QUERIES_EMBEDDING_FILENAME}")

        self.name = data_path.name
        self.dataframe = pd.read_csv(data_path / self.QUERIES_DF_FILENAME, index_col='qid')
        self.embeddings = np.load(data_path / self.QUERIES_EMBEDDING_FILENAME)
        self.dataframe['query'] = self.dataframe['query'].astype(str)

        if Path(data_path / self.SPACY_DIR).exists():
            self.spacy_index = SpacyIndex(data_path / self.SPACY_DIR)
        else:
            self.spacy_index = None


    def get_query_text(self, query_id) -> str:
        return str(self.dataframe.loc[query_id].query)

    def get_query_embedding(self, query_id) -> np.ndarray:
        return self.embeddings[self.dataframe.loc[query_id].embedding_index]

    def get_query_embeddings(self, query_ids) -> np.ndarray:
        return self.embeddings[self.dataframe.loc[query_ids].embedding_index.values]

    def get_all_query_ids_and_embeddings(self) -> Tuple[List[Union[int, str]], np.ndarray]:
        """To search with faiss and efficiently build ranking"""
        return (
            self.dataframe.index.values,
            self.embeddings[self.dataframe.embedding_index.values]
        )

    def iter_queries(self) -> Generator[Tuple[Union[int, str], str], None, None]:
        return self.dataframe['query'].items()

    def num_queries(self) -> int: 
        return self.dataframe.shape[0]

    @classmethod
    def create_from_dataframe(cls, path: Path, name: str, 
                            sbert_embedder: SentenceTransformer, input_dataframe: pd.DataFrame, 
                            id_field_name: str='qid', text_field_name: str='query'):
        path = Path(path)
        qset_path = path / name
        qset_path.mkdir(exist_ok=False)

        queries = input_dataframe[[id_field_name, text_field_name]].drop_duplicates(subset=id_field_name) \
            .rename(columns={id_field_name: 'qid', text_field_name: 'query'}) \
            .set_index('qid')
        queries['embedding_index'] = range(queries.shape[0])
        queries.to_csv(qset_path / cls.QUERIES_DF_FILENAME)

        np.save(qset_path / cls.QUERIES_EMBEDDING_FILENAME, 
                sbert_embedder.encode(list(map(str, queries['query'].values)), normalize_embeddings=True))

        return cls(qset_path)
    

class Ranking:
    def __init__(self, query_id: Union[int, str], dataframe: pd.DataFrame):
        self.df = dataframe.set_index('rank')
        self.query_id = query_id

    def get_document_ids(self, k: int=None) -> list:
        if k is None:
            return self.df.result_id.values
        else:
            return self.df.loc[range(1, k+1)].result_id

    def iter_results_with_scores(self) -> Generator[Tuple[Union[int, str], float], None, None]:
        yield from self.df[['result_id', 'score']].itertuples(index=False)

    def display(self, docset: DocumentSet, queryset: QuerySet, display_fn=print):
        display_fn(f"Query: {queryset.get_query_text(self.query_id)}")
        display_fn(self.df.assign(text=docset.get_document_texts(self.df.result_id.values)))


class RankingSet:
    def __init__(self, 
                df: pd.DataFrame = None,
                df_path = None):
        if (df is None and df_path is None) or \
            (df is not None and df_path is not None):
            raise ValueError("Must specify either df or df_path")

        
        if df is not None:
            self.dataframe = df
        elif df_path is not None:
            self.dataframe = pd.read_csv(df_path)

        assert [col in self.dataframe.columns
                for col in ['qid', 'rank', 'result_id', 'score']]

    def saved(self, path) -> 'RankingSet':
        self.dataframe.to_csv(path, index=False)
        return self

    def get_query_ids(self) -> List[Union[str, int]]:
        return self.dataframe.qid.drop_duplicates().values

    def get_ranking(self, query_id) -> Ranking:
        return Ranking(query_id, self.dataframe[self.dataframe['qid'] == query_id])

    def iter_rankings(self) -> Generator[Ranking, None, None]:
        yield from starmap(lambda qid, ranking_df: Ranking(qid, ranking_df),
                            self.dataframe.groupby('qid').__iter__())

    def iter_document_findings(self, doc_id, k=None) -> Generator[Tuple[Union[str, int], float], None, None]:
        """Given a docid, find every ranking that the document appears in."""
        if 'result_id' in self.dataframe._get_numeric_data().columns:
            query = f'result_id == {doc_id}'
        else:
            query = f'result_id == \"{doc_id}\"'
        
        if k is not None:
            yield from self.dataframe.query(f'rank <= {k} and {query}').itertuples()
        else:
            yield from self.dataframe.query(query).itertuples()

    def num_rankings(self) -> int:
        return self.dataframe.qid.nunique()

    def median_rank_score_diff(self, topk=20):
        """To determine whether a score difference would (probably) make for a difference in rank."""
        return np.median(self.dataframe.groupby('qid').filter(lambda df: len(df) > 1).groupby('qid').head(-1).query(f'rank < {topk}').score.values - 
                        self.dataframe.groupby('qid').filter(lambda df: len(df) > 1).query(f'rank > 1 and rank <= {topk}').score.values)

    def get_score(self, qid, result_id):
        result_row = self.dataframe.query(f"qid == {qid} and result_id == {result_id}")
        if result_row.shape[0] != 1: 
            raise ValueError(f"Error: Found {result_row.shape[0]} entries for qid={qid}, result_id={result_id}. Expcted 1.")
        return result_row.iloc[0].score

    @classmethod
    def accumulate(cls, rankings: Generator[Ranking, None, None]) -> 'RankingSet':
        return cls(df=pd.concat(map(lambda ranking: ranking.df.reset_index(), rankings))
                        [['qid', 'rank', 'result_id', 'score']].sort_values(by=['qid', 'rank']))
    
