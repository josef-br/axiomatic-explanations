"""Algorithms that take a diagnostic dataset and a ranking model and calculate the scorings for each sample in the dataset"""

from typing import Iterable, Tuple, Union
from .diagnostic import DiagnosticDataset
from sentence_transformers import SentenceTransformer
import torch
from itertools import starmap
from more_itertools import chunked
from .get_data import Data
import pandas as pd
import logging
import numpy as np
from tqdm.autonotebook import tqdm

class Scorer:
    def __init__(self, name:str, lookup_known=False):
        self.name = name
        self._lookup_known = lookup_known

    def score_dataset(self, diag_data: DiagnosticDataset, dataobj: Data, show_progress_bar=True):
        scored_df = self._create_scores(diag_data, dataobj, show_progress_bar)
        self._expand_and_save_scores(scored_df, diag_data, dataobj)

    def _create_scores(self, diag_data: DiagnosticDataset, dataobj: Data, show_progress_bar):
        if self._lookup_known and diag_data.df['doc_a_id'].notna().all():
            doc_a_scores = self._score_known_pairs(tqdm(diag_data.df[['qid', 'doc_a_id']].itertuples(index=False), 
                                                    total=diag_data.df.shape[0], 
                                                    desc=f'{diag_data.qset_name} {diag_data.probe_name} Document A scoring', 
                                                    disable=not show_progress_bar),
                                                dataobj, diag_data.qset_name)
        else:
            doc_a_scores = self._score_unknown_pairs(tqdm(diag_data.df[['qid', 'doc_a_text']].itertuples(index=False), 
                                                    total=diag_data.df.shape[0],
                                                    desc=f'{diag_data.qset_name} {diag_data.probe_name} Document A scoring', 
                                                    disable=not show_progress_bar),
                                                dataobj, diag_data.qset_name)
        
        if self._lookup_known and diag_data.df['doc_b_id'].notna().all():
            doc_b_scores = self._score_known_pairs(tqdm(diag_data.df[['qid', 'doc_b_id']].itertuples(index=False), 
                                                        total=diag_data.df.shape[0],
                                                        desc=f'{diag_data.qset_name} {diag_data.probe_name} Document B scoring', 
                                                        disable=not show_progress_bar),
                                                    dataobj, diag_data.qset_name)
        else:
            doc_b_scores = self._score_unknown_pairs(tqdm(diag_data.df[['qid', 'doc_b_text']].itertuples(index=False), 
                                                        total=diag_data.df.shape[0],
                                                        desc=f'{diag_data.qset_name} {diag_data.probe_name} Document B scoring', 
                                                        disable=not show_progress_bar),
                                                    dataobj, diag_data.qset_name)

        return pd.DataFrame({'a_score': doc_a_scores, 'b_score': doc_b_scores}, index=diag_data.df.index)


    def _expand_and_save_scores(self, scored_df: pd.DataFrame, diag_data: DiagnosticDataset, dataobj: Data):
        scored_df['score_diff'] = scored_df['b_score'] - scored_df['a_score']

        ranking_name = f'{self.name}_{diag_data.qset_name}.csv'
        if ranking_name in dataobj.list_rankingsets():
            score_delta = dataobj.get_rankings(diag_data.qset_name, self.name).median_rank_score_diff()
            scored_df['rank_diff'] =  np.select(condlist=[scored_df.score_diff > score_delta, scored_df.score_diff < -1*score_delta],
                                                choicelist=[1, -1],
                                                default=0)
        else:
            logging.warn(f"Found no ranking for qset {diag_data.qset_name} and method {self.name}. Result is incomplete (Fix this manually)")

        result_path = dataobj.path / 'probes' / diag_data.qset_name / self.name
        result_path.mkdir(exist_ok=True)
        scored_df.to_csv(result_path / diag_data.probe_name, index=True)

    
    def _score_known_pairs(self, qid_docid_pairs: Iterable[Tuple[int, Union[int, str]]], dataobj: Data, qset_name: str) -> Iterable[float]:
        raise NotImplementedError("This is an abstract class")

    
    def _score_unknown_pairs(self, qid_doc_pairs: Iterable[Tuple[int, str]], dataobj: Data, qset_name: str) -> Iterable[float]:
        raise NotImplementedError("This is an abstract class")

class NeuralScorer(Scorer):
    def __init__(self, model_base, batch_size=256, internal_batch_size=32):
        super().__init__('neural', False)
        self.model = SentenceTransformer(model_base, cache_folder='__model_cache__')
        self.batch_size = batch_size
        self.internal_batch_size = internal_batch_size

    def _score_known_pairs(self, qid_docid_pairs: Iterable[Tuple[int, Union[int, str]]], dataobj: Data, qset_name: str) -> Iterable[float]:
        # TODO make this more efficient?? at the moment simply embedding every text is faster than looking them all up from FAISS index (at least on GPU)
        faiss_index = dataobj.get_faiss_index()
        qset = dataobj.get_queryset(qset_name)
        return starmap(lambda qid, docid: (qset.get_query_embedding(qid) * faiss_index.get_document_embedding(docid)).sum(), qid_docid_pairs)

    def _score_unknown_pairs(self, qid_doc_pairs: Iterable[Tuple[int, str]], dataobj: Data, qset_name: str) -> Iterable[float]:
        qset = dataobj.get_queryset(qset_name)
        with torch.no_grad():
            for batch in chunked(qid_doc_pairs, self.batch_size):
                # load embedded queries
                queries_embed = qset.get_query_embeddings(list(map(lambda p: p[0], batch)))
                docs_embed = self.model.encode(list(map(lambda p: p[1], batch)),
                                                normalize_embeddings=True, batch_size=self.internal_batch_size,
                                                show_progress_bar=False)


                yield from (queries_embed * docs_embed).sum(axis=1).tolist()

class BM25Scorer(Scorer):
    def __init__(self, field_name='text_stopped'):
        super().__init__('bm25', True)
        self.field_name = field_name

    def _score_known_pairs(self, qid_docid_pairs: Iterable[Tuple[int, Union[int, str]]], dataobj: Data, qset_name: str) -> Iterable[float]:
        l_index = dataobj.get_lucene_index()
        qset = dataobj.get_queryset(qset_name)
        results = list(starmap(lambda qid, docid: l_index.score_bm25_known(qset.get_query_text(qid), docid, self.field_name), qid_docid_pairs))
        return results

    def _score_unknown_pairs(self, qid_doc_pairs: Iterable[Tuple[int, str]], dataobj: Data, qset_name: str) -> Iterable[float]:
        l_index = dataobj.get_lucene_index()
        qset = dataobj.get_queryset(qset_name)
        results = list(starmap(lambda qid, doctext: l_index.score_bm25_unknown(qset.get_query_text(qid), str(doctext), self.field_name), qid_doc_pairs))
        return results

    def _create_scores(self, diag_data: DiagnosticDataset, dataobj: Data, show_progress_bar):
        return super()._create_scores(diag_data, dataobj, show_progress_bar) \
                        .query('a_score > 0.0 or b_score > 0.0')