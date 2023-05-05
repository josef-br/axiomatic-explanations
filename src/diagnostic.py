"""Basic classes to create and evaluate diagnostic datasets"""
from dataclasses import dataclass
import pandas as pd
from .dataset import DocumentSet, QuerySet, RankingSet
from .get_data import Data
from typing import Generator
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import math


@dataclass
class ProbeSample:
    # one instance in a diagnostic dataset
    qid: str
    doc_a_id: str
    doc_b_id: str
    query_text: str
    doc_a_text: str
    doc_b_text: str
    metadata: dict


class Prober:
    """Abstract class for all algorithms that sample a diagnostic dataset"""
    def __init__(self, name: str):
        self.name = name

    def _generate_probes(self, qset: QuerySet, rset: RankingSet, docset: DocumentSet, show_progress_bar=True) -> Generator[ProbeSample, None, None]:
        raise NotImplementedError("This is only an abstract class")

    def build_diagnostic_dataset(self, dataobj: Data, qset_name: str, force_overwrite=False, show_progress_bar=True):
        if qset_name not in dataobj.list_qsets():
            raise ValueError(f"Could not find queryset '{qset_name}' in {str(dataobj.path)}")

        if f'{self.name}.csv' in dataobj.list_probes(qset_name) and not force_overwrite:
            logging.info(f'Probe {self.name} already exists for {qset_name} and overwrite is disabled.')
            return

        rankingset = dataobj.get_rankings(qset_name, 'neural') # TODO extend this and also use BM25 Ranking
        sample_generator = self._generate_probes(dataobj.get_queryset(qset_name), rankingset, dataobj.get_docset(), show_progress_bar)

        diag_dataset = pd.DataFrame.from_records(map(lambda probe: probe.__dict__, sample_generator))
        diag_dataset.index.set_names('index', inplace=True)
         # metadata is a dict, explode it into extra columns
        diag_dataset = diag_dataset.join(pd.json_normalize(diag_dataset.get('metadata', {}), max_level=0)).drop('metadata', axis='columns', errors='ignore')

        dataset_dir = dataobj.path / f'probes/{qset_name}'
        dataset_dir.mkdir(exist_ok=True, parents=True)
        diag_dataset.to_csv(dataset_dir / f'{self.name}.csv', index=True)

        logging.info(f"Wrote {diag_dataset.shape[0]} samples to new diganostic dataset {qset_name}/{self.name}")
        return diag_dataset


class DiagnosticDataset:
    def __init__(self, dataobj: Data, qset_name:str, probe_name:str):
        if not probe_name.endswith('.csv'):
            probe_name = probe_name + '.csv'

        df_path = dataobj.path / 'probes' / qset_name / probe_name
        if not df_path.exists():
            raise FileNotFoundError(f"No probe {qset_name}/{probe_name} could be found")
        self.df = pd.read_csv(df_path, index_col='index')
        self.qset_name = qset_name
        self.probe_name = probe_name

        self.path = df_path
        self.df[['query_text', 'doc_a_text', 'doc_b_text']] = self.df[['query_text', 'doc_a_text', 'doc_b_text']].astype('string')

    def has_scored(self, model):
        return Path(self.path.parent / model / self.probe_name).exists()

    def scored(self, model):
        assert self.has_scored(model), f"Probe {self.qset_name}/{self.probe_name} is not scored for {model}"
        return ScoredDiagnosticData(self, model=model)
        

class ScoredDiagnosticData:
    def __init__(self, diag_data: DiagnosticDataset, model: str=None, result_df=None):
        if model is None and result_df is None or model is not None and result_df is not None:
            raise ValueError("Must specify either a location to the df or a result df")
        if result_df is None:
            result_df_path = Path(diag_data.path.parent / model / diag_data.probe_name)
            if not result_df_path.exists():
                raise FileNotFoundError(f"No results for {model}-scoring on {result_df_path.stem} dataset was found")

            self.result_df = pd.read_csv(result_df_path, index_col='index')
        else:
            self.result_df = result_df

        self.diag_data = diag_data

        if not all([result_id in self.diag_data.df.index for result_id in self.result_df.index]):
            raise ValueError(f"The set of indices between diagnostic dataset and result do not match. Are they the same version?")

    def abnirml_score(self):
        return self.result_df['rank_diff'].mean()
    
    def kendall_tau_b(self):
        pos = (self.result_df['rank_diff'] == 1).sum()
        neg = (self.result_df['rank_diff'] == -1).sum()
        return (pos - neg) / math.sqrt((pos + neg) * (self.result_df.shape[0]))

    def is_significant(self, alpha=.01):
        # two-sided paired t-test, just like the ABNIRML friends
        return ttest_rel(self.result_df['a_score'], self.result_df['b_score']).pvalue < alpha

    def dataframe(self, columns=['query_text', 'doc_a_text', 'doc_b_text', 'a_score', 'b_score', 'score_diff', 'rank_diff']):
        return self.diag_data.df.join(self.result_df, how='inner')[columns]
    
    def query(self, query):
        q_index = self.diag_data.df.query(query).index
        return ScoredDiagnosticData(self.diag_data, result_df=self.result_df.loc[self.result_df.index.intersection(q_index)])
