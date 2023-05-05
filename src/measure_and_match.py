"""Implementation of the measure-and-match method. 
Basically there are a few classes that define measureable features, and then an algorithm that creates a dataset 
by holding some features constant and one variable."""
from .lucene import LuceneIndex
from .diagnostic import Prober, ProbeSample
from .dataset import DocumentSet, QuerySet, Ranking, RankingSet
from itertools import starmap, permutations, chain
from more_itertools import pairwise
from operator import xor
from tqdm.autonotebook import tqdm
from typing import List, Generator

# Shamelessly stolen and then mutilated beyond recognition from the ABNIRML implementation

class ConstVar:
    def __init__(self, name: str, index: LuceneIndex, epsilon=0, field_name='text'):
        self.name = name
        self.index = index
        self.epsilon = abs(epsilon) # epsilon is strictly positive or 0
        self.field_name = field_name

    def is_valid(self, value):
        return value > 0
    
    def score(self, query, doc_id):
        raise NotImplementedError

    def is_const(self, a, b):
        return abs(a - b) <= self.epsilon

    def is_var(self, a, b):
        return a - self.epsilon > b


class Len(ConstVar):
    def __init__(self, index: LuceneIndex, epsilon=0, field_name='text', **kwargs):
        super().__init__('len', index, epsilon, field_name)

    def score(self, query, doc_id):
        """This one is independent of the query"""
        return self.index.doclen(doc_id, self.field_name)

    """Const and Var are either absolute (when >=1.0)
    or relative with the larger doc's size (when between 0.0 and 1.0"""
    def is_const(self, a, b):
        if a == 0 and b == 0: return True
        elif self.epsilon == 0.: return a==b
        elif self.epsilon >= 1.: return abs(a - b) <= self.epsilon
        elif self.epsilon < 1.: return 1. - min(a, b) / max(a, b) <= self.epsilon

    def is_var(self, a, b):
        return not self.is_const(a, b) and a > b


class Tf(ConstVar):
    def __init__(self, index: LuceneIndex, epsilon=0, field_name='text', **kwargs):
        super().__init__('tf', index, epsilon, field_name)
        self._prev_query = None # for caching
        self._prev_qtoks = None

    def score(self, query, doc_id):
        if query != self._prev_query:
            self._prev_query = query
            self._prev_qtoks = self.index.tokenize(self._prev_query, self.field_name)
        return self.index.tfs(doc_id, self._prev_qtoks, self.field_name)

    def is_valid(self, value):
        return any(map(lambda v: v > 0, value.values()))

    def is_const(self, a, b):
        # all are equal or deviate at most by epsilon
        return all(abs(a.get(t, 0) - b.get(t, 0)) <= self.epsilon for t in a.keys() | b.keys())

    def is_var(self, a, b):
        # all are at least as large, and one is strictly larger
        if all(map(lambda tf: tf==0, a)) or all(map(lambda tf: tf==0, b)): # don't count no tf matches at all
            return False
        return (all(a.get(t, 0) - self.epsilon >= b.get(t, 0) for t in a.keys() | b.keys()) and \
               any(a.get(t, 0) - self.epsilon > b.get(t, 0) for t in a.keys() | b.keys()))


class BoolTf(ConstVar):
    def __init__(self, index: LuceneIndex, epsilon=0, field_name='text', **kwargs):
        super().__init__('booltf', index, epsilon, field_name)
        self._prev_query = None
        self._prev_qtoks = None

    def score(self, query, doc_id):
        if query != self._prev_query:
            self._prev_query = query
            self._prev_qtoks = self.index.tokenize(self._prev_query, self.field_name)
        return {term: tf > 0 for term, tf in self.index.tfs(doc_id, self._prev_qtoks, self.field_name).items()}

    def is_valid(self, value):
        return any(value.values())

    def is_const(self, a, b):
        # at most epsilon are inequal
        return sum([a[key] != b[key] for key in a.keys() | b.keys()]) <= self.epsilon

    def is_var(self, a, b):
        # one contains all terms of the other plus at least epsilon+1 more
        if not any(a.values()) or not any(b.values()): # don't count no tf matches at all
            return False
        # in sum a has more than b and there is no term that is in b but not in a
        return (sum(a.values()) - sum(b.values()) > self.epsilon) and not any(not a[t] and b[t]  for t in a.keys() | b.keys())


class SumTf(ConstVar):
    def __init__(self, index: LuceneIndex, epsilon=0, field_name='text', **kwargs):
        super().__init__('sumtf', index, epsilon, field_name)
        self._prev_query = None
        self._prev_qtoks = None

    def score(self, query, doc_id):
        if query != self._prev_query:
            self._prev_query = query
            self._prev_qtoks = self.index.tokenize(self._prev_query, self.field_name)
        return sum(self.index.tfs(doc_id, self._prev_qtoks, self.field_name).values())


class Proximity(ConstVar):
    """For queries with at least two terms: Measure the smallest distance between two query terms"""
    def __init__(self, index: LuceneIndex, epsilon=0, field_name='text', **kwargs):
        super().__init__('proximity', index, epsilon, field_name)
        self._prev_query = None # for caching
        self._prev_qtoks = None

    def score(self, query, doc_id):
        if query != self._prev_query:
            self._prev_query = query
            self._prev_qtoks = self.index.tokenize(self._prev_query, self.field_name)
        
        positions = self.index.positions(doc_id, self._prev_qtoks, self.field_name)
        positions = {
            term: pos_array
            for term, pos_array in positions.items()
            if len(pos_array) > 0
        }

        if len(positions) < 2:
            return 0
        
        all_positions = sorted(chain.from_iterable(positions.values()))
        pairwise_distances = [b-a for a, b in pairwise(all_positions)]

        return min(pairwise_distances)


class QueryTermOrdering(ConstVar):
    """This is a predicate. Either all query terms appear in the document in order (1) or not (-1). 
    Also 0 for invalid items (only one or no matching terms)"""
    def __init__(self, index: LuceneIndex, epsilon=0, field_name='text', **kwargs):
        super().__init__('termordering', index, epsilon, field_name)
        self._prev_query = None # for caching
        self._prev_qtoks = None

    def score(self, query, doc_id):
        if query != self._prev_query:
            self._prev_query = query
            self._prev_qtoks = self.index.tokenize(self._prev_query, self.field_name)
        
        positions = self.index.positions(doc_id, self._prev_qtoks, self.field_name)
        positions = {
            term: pos_array
            for term, pos_array in positions.items()
            if len(pos_array) > 0
        }

        if len(positions) < 2:
            return 0
        # best algorithm I could come up with: Iterate over query terms that are int he document (ordered),
        # for each one remember the smallest next possible position.
        # If the next query term has no position larger than the current one, return -1 (=False)
        last_qterm_position = -1
        for query_term in [q_tok for q_tok in self._prev_qtoks if q_tok in positions.keys()]:
            qterm_positions = positions[query_term]

            larger_positions = list(filter(lambda pos: pos > last_qterm_position, qterm_positions))
            if len(larger_positions) == 0:
                return -1
            last_qterm_position = min(larger_positions)

        # If we went through all query terms and always found a valid successor, the doc contains the terms in order
        return 1 # =True

    def is_valid(self, value):
        return value != 0

    def is_const(self, a, b):
        return a == b    

class Idf(ConstVar):
    """Value is a dict of all matching query terms and their IDF (qterm=key, idf=value).
    To be a comparable var, two docs have to have different sets of matching terms, but all that are in d1 but not in d2 have to have higher idf"""
    def __init__(self, index: LuceneIndex, epsilon=0, field_name='text', **kwargs):
        super().__init__('idf', index, epsilon, field_name)
        self._prev_query = None # for caching
        self._prev_qtoks = None

    def score(self, query, doc_id):
        if query != self._prev_query:
            self._prev_query = query
            self._prev_qtoks = self.index.tokenize(self._prev_query, self.field_name)
        return {term: self.index.idf(term, self.field_name) 
                for term, tf in self.index.tfs(doc_id, self._prev_qtoks, self.field_name).items()
                if tf > 0}
    
    def is_valid(self, value):
        return bool(value)

    def is_const(self, a, b):
        return a.keys() == b.keys()

    def is_var(self, a, b):
        only_in_d1 = set([term for term in a.keys() if term not in b.keys()])
        only_in_d2 = set([term for term in b.keys() if term not in a.keys()])
        return len(only_in_d1) > 0 and \
                len(only_in_d2) > 0 and \
                all([a[t1] >= b[t2]
                     for t1 in only_in_d1
                     for t2 in only_in_d2])


class ExactMatch(ConstVar):
    """Measures which terms are exactly contained in the document. 
    It returns a dictionary of all terms that match stemmed, with a bool-value whether they also match exact."""
    def __init__(self, index: LuceneIndex, epsilon=0, field_name='text', **kwargs):
        super().__init__('exactmatch', index, epsilon, field_name)
        self._prev_query = None # for caching
        self._prev_qtoks = None
        self._prev_qtoks_raw = None

    def score(self, query, doc_id):
        # difference to Tf: Tokenize and match the query with the '_raw'-Field -> no stemming was performed
        if query != self._prev_query:
            self._prev_query = query
            self._prev_qtoks = self.index.tokenize(self._prev_query, self.field_name)
            self._prev_qtoks_raw = self.index.tokenize(self._prev_query, f'{self.field_name}_raw')
            assert len(self._prev_qtoks) == len(self._prev_qtoks_raw) # if not something went wrong...
        tfs_stemmed = self.index.tfs(doc_id, self._prev_qtoks, field_name='text') # if stopword tokens should be ignored they were removed during query tokenization
        tfs_raw = self.index.tfs(doc_id, self._prev_qtoks_raw, field_name='text_raw')
        return { # dict of all terms that are contained in a stemmed version. bool-value determines if they are also contained exactly
            raw_term: tfs_raw[raw_term] > 0
            for raw_term, stemmed_term in zip(self._prev_qtoks_raw, self._prev_qtoks)
            if tfs_stemmed[stemmed_term] > 0
        }
        
    def is_valid(self, value):
        # valid when there are stemmed-matching terms, hence dict is not empty
        return bool(value)

    def is_const(self, a, b):
        # number of deviating items is at most epsilon
        return sum([xor(a.get(term, False), b.get(term, False)) for term in a.keys() | b.keys()]) <= self.epsilon

    def is_var(self, a, b):
        # to be var all true values from the 'lower' item must also be true in the 'higher', plus the 'higher' must have more exact matches
        return (all([a.get(term, False) or not b.get(term, False) for term in a.keys() | b.keys()]) and sum(a.values())) - self.epsilon > sum(b.values())
    

class ConstVarProber(Prober):
    def __init__(self, name: str,
                index: LuceneIndex, 
                var: ConstVar.__class__, 
                const: List[ConstVar.__class__],
                var_epsilon=0.,
                const_epsilons=None,
                field_name='text',
                var_kwargs={},
                const_kwargs=None):
        super().__init__(name)
        
        if const_epsilons is None:
            const_epsilons = [0.] * len(const)
        if const_kwargs is None:
            const_kwargs = [{}] * len(const)

        assert len(const) == len(const_epsilons)

        self.var = var(index, var_epsilon, field_name, **var_kwargs)
        self.const = [c(index, ep, field_name, **kwargs) for c, ep, kwargs in zip(const, const_epsilons, const_kwargs)]

        self.index = index

    def _is_valid_doc_pair(self, doc_1, doc_2):
        return self.var.is_valid(doc_1['var_score']) and \
                self.var.is_valid(doc_2['var_score']) and \
                self.var.is_var(doc_2['var_score'], doc_1['var_score']) and \
                all(starmap(lambda c, score1, score2: c.is_const(score1, score2), 
                            zip(self.const, doc_1['const_scores'], doc_2['const_scores'])))

    def _generate_probes(self, qset: QuerySet, rset: RankingSet, docset: DocumentSet, show_progress_bar=True) -> Generator[ProbeSample, None, None]:
        ranking: Ranking
        for ranking in tqdm(rset.iter_rankings(), total=rset.num_rankings(), disable=not show_progress_bar, desc=f"{qset.name}/{self.name}"):
            query_text = qset.get_query_text(ranking.query_id)
            doc_scores = [{
                'id': result_id,
                'var_score': self.var.score(query_text, result_id),
                'const_scores': [c.score(query_text, result_id) for c in self.const]
            } for result_id in ranking.get_document_ids()]

            for doc_1, doc_2 in permutations(doc_scores, 2):
                if self._is_valid_doc_pair(doc_1, doc_2):
                    yield ProbeSample(qid=ranking.query_id, doc_a_id=doc_1['id'], doc_b_id=doc_2['id'],
                                    query_text=query_text, 
                                    doc_a_text=docset.get_document_text(doc_1['id']),
                                    doc_b_text=docset.get_document_text(doc_2['id']),
                                    metadata=dict({
                                                f'a_var_{self.var.name}': doc_1['var_score'],
                                                f'b_var_{self.var.name}': doc_2['var_score'],
                                            }, **{
                                                f'a_const_{c.name}': doc_1['const_scores'][idx]
                                                for idx, c in enumerate(self.const)
                                            }, **{
                                                f'b_const_{c.name}': doc_2['const_scores'][idx]
                                                for idx, c in enumerate(self.const)
                                            }))