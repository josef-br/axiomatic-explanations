"""Methods to collect diagnostic datasets via dataset transfer.
Actually there is only one in this thesis, namely the text succinctness diagnostic"""

from .diagnostic import Prober, ProbeSample
from .dataset import DocumentSet, QuerySet, RankingSet
from .get_data import CqaDupStackData, Data
import pandas as pd
from tqdm.autonotebook import tqdm
from typing import Generator

class TextSuccintnessProber(Prober):
    """Additional level of abstraction, since this has to be implemented separately for different datasources"""
    def __init__(self, dataobj: Data):
        super().__init__('succinctness')
        self.succinctness_pairs = self._build_succinctness_pairs(dataobj)

    def _build_succinctness_pairs(self, dataobj: Data) -> pd.DataFrame: pass # return a dataframe with three cols: <short>, <long>, <doc_ids>

    def _generate_probes(self, qset: QuerySet, rset: RankingSet, docset: DocumentSet, show_progress_bar=True) -> Generator[ProbeSample, None, None]:
        # iter through succinctness pairs; for each find all queries that found one of the docs
        for pair in tqdm(self.succinctness_pairs.itertuples(), total=self.succinctness_pairs.shape[0], disable=not show_progress_bar):
            matching_qids = set(rset.dataframe.query(f"result_id in {pair.doc_ids}").qid.values)
            for qid in matching_qids:
                yield ProbeSample(qid, None, None, 
                                qset.get_query_text(qid), pair.short, pair.long, 
                                {})



class CQATextSuccintnessProber(TextSuccintnessProber):
    def __init__(self, dataobj: CqaDupStackData):
        super().__init__(dataobj)

    def _build_succinctness_pairs(self, dataobj: CqaDupStackData) -> pd.DataFrame:
        """The CQA Dataframe ist built so that the title and body of one question are two separate documents.
        They have the same id with a postfix 't' or 'b'. To obtain succinctness pairs we pivot the two back together."""
        pivoted_df = dataobj.get_docset().dataframe.reset_index() \
                            .assign(orig_id=lambda df: df.doc_id.str.slice(0, -1)) \
                            .pivot(index='orig_id', columns=['variable'], values=['value', 'doc_id'])

        return pivoted_df.value.rename(columns={'text': 'long', 'title': 'short'}) \
                .assign(doc_ids=pivoted_df.doc_id.apply(list, axis=1)) \
                .query("~long.isna()") # sometimes the long documents were omitted due to length, in this case no query pair can be built

    