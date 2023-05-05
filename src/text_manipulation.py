"""The different methods to collect diagnostic datasets via text manipulation"""

from .diagnostic import Prober, ProbeSample
from .lucene import LuceneIndex
from .dataset import DocumentSet, QuerySet, RankingSet
from .spacy import SpacyIndex
from .util import TOKEN_PATTERN_ONLY_ALPHANUM
from tqdm.autonotebook import tqdm
from typing import Generator, List
import re
import pandas as pd
import logging

class SpellErrorProber(Prober):
    def __init__(self, index: LuceneIndex, max_samples_per_term=20, typos_tsv_path='resources/wiki_typos.tsv'):
        super().__init__('spellerror')
        self.index = index
        self.typos = pd.read_csv(typos_tsv_path, delimiter='\t', header=None, names=['typo', 'correct'])
        # they are comma-separated, so...
        self.typos['correct'] = self.typos.correct.str.split(',')
        self.typos = self.typos.explode('correct')
        self.sample_size = max_samples_per_term

    def _generate_probes(self, qset: QuerySet, rset: RankingSet, docset: DocumentSet, show_progress_bar=True) -> Generator[ProbeSample, None, None]:
        """
        Strategy: 
        1. Loop over typos; find all documents that contain the original via inverted index
        2. Find all queries that find this doc; create a probe where the original is replaced by the spelling error
        """
        for entry in tqdm(self.typos.itertuples(), total=self.typos.shape[0], disable=not show_progress_bar, desc=f"{qset.name}/{self.name}"):
            orig_term_pattern = re.compile(re.escape(entry.correct), re.IGNORECASE)
            orig_idf = self.index.idf(entry.correct, field_name='text_raw') # TODO vllt idf vom stemmed text ?!

            sampler = self.index.sample_documents_containing(entry.correct, field_name='text_raw')

            for doc_id, doc_text in sampler:                    
                # replace correct string with typo, ignoring casing
                mod_text = orig_term_pattern.sub(entry.typo, doc_text)

                yield from map(lambda r: ProbeSample(qid=r.qid, doc_a_id=doc_id, doc_b_id=None,
                                                    query_text=qset.get_query_text(r.qid),
                                                    doc_a_text=doc_text, doc_b_text=mod_text,
                                                    metadata={
                                                        'original_term': entry.correct,
                                                        'modified_term': entry.typo,
                                                        'term_is_in_query': entry.correct.lower() in qset.get_query_text(r.qid).lower(),
                                                        'term_idf': orig_idf
                                                    }),
                                    rset.iter_document_findings(doc_id))


class LemmatizationProber(Prober):
    OPEN_CLASS_POS = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB'] # see https://universaldependencies.org/u/pos/

    def __init__(self, spacy_index: SpacyIndex):
        super().__init__('lemmatize')
        self.sp_index = spacy_index

    def _generate_probes(self, qset: QuerySet, rset: RankingSet, docset: DocumentSet, show_progress_bar=True) -> Generator[ProbeSample, None, None]:
        """Simple: Just lemmatize each document text. Save the number of changed tokens as metadata"""
        for r in tqdm(rset.iter_rankings(), total=rset.num_rankings(), disable=not show_progress_bar, desc=f"{qset.name}/{self.name}"):
            for doc_id, score in r.iter_results_with_scores():
                nlp_doc = self.sp_index.get_nlp(doc_id)
                lemmatized_tokens = [token.lemma_ if token.pos_ in self.OPEN_CLASS_POS else token.text for token in nlp_doc]
                lemmatized_text = ''.join([l_token + nlp_token.whitespace_ for l_token, nlp_token in zip(lemmatized_tokens, nlp_doc)])
                changed_terms = sum([nlp_tok.text.lower() != l_tok.lower() for nlp_tok, l_tok in zip(nlp_doc, lemmatized_tokens)])
                if changed_terms > 0: # do not create probes where the lemmatization leaves the document unchanged
                    yield ProbeSample(qid=r.query_id, doc_a_id=doc_id, doc_b_id=None,
                                        query_text=qset.get_query_text(r.query_id),
                                        doc_a_text=nlp_doc.text, doc_b_text=lemmatized_text,
                                        metadata={
                                            'doc_len': len(nlp_doc),
                                            'num_changed_terms': changed_terms,
                                            'ratio_changed': changed_terms / len(nlp_doc)
                                    })

class StopwordRemoveProber(Prober):
    def __init__(self, stopword_list: List[str], tokenizer_fn=None):
        super().__init__('stopwords')
        if tokenizer_fn is None:
            self.tokenizer_fn = re.compile(TOKEN_PATTERN_ONLY_ALPHANUM).findall
        else:
            self.tokenizer_fn = tokenizer_fn

        self.stopwords = stopword_list

    def _generate_probes(self, qset: QuerySet, rset: RankingSet, docset: DocumentSet, show_progress_bar=True) -> Generator[ProbeSample, None, None]:
        """Tokenize the document text and remove all the stopwords"""
        for r in tqdm(rset.iter_rankings(), total=rset.num_rankings(), disable=not show_progress_bar, desc=f"{qset.name}/{self.name}"):
            for doc_id, score in r.iter_results_with_scores():
                doc_text = docset.get_document_text(doc_id, docset.default_text_field_name)
                tokenized_doc = self.tokenizer_fn(doc_text)
                unstopped_doc = [tok for tok in tokenized_doc if tok.lower() not in self.stopwords]
                if len(tokenized_doc) != len(unstopped_doc) and len(unstopped_doc) > 0: # do not create probes when there are no stopwords to remove
                    yield ProbeSample(qid=r.query_id, doc_a_id=doc_id, doc_b_id=None,
                                        query_text=qset.get_query_text(r.query_id),
                                        doc_a_text=doc_text, doc_b_text=' '.join(unstopped_doc),
                                        metadata={
                                            'doc_len': len(tokenized_doc),
                                            'removed_terms': len(tokenized_doc) - len(unstopped_doc)
                                        })


class AbbreviationProber(Prober):
    def __init__(self, docs_spacy_index, abbreviation_df_loc='resources/abbreviations.csv'):
        super().__init__('thesaurus')
        self.docs_spacy = docs_spacy_index
        self.abbrev_df = pd.read_csv(abbreviation_df_loc)

    def _generate_probes(self, qset: QuerySet, rset: RankingSet, docset: DocumentSet, show_progress_bar=True) -> Generator[ProbeSample, None, None]:
        """Analyze the document with spacy. For each nound, verb oder adjective check if there is a thesaurus entry.
        If so, choose the most similar thesaurus entry and yield one probe per replaced term."""
        if not qset.spacy_index:
            logging.error("No spacy index in qset. Cannot generate data")
            return
        qset_spacy = qset.spacy_index
        for r in tqdm(rset.iter_rankings(), total=rset.num_rankings(), disable=not show_progress_bar, desc=f"{qset.name}/{self.name}"):
            nlp_query = qset_spacy.get_nlp(r.query_id)
            for doc_id, score in r.iter_results_with_scores():
                nlp_doc = self.docs_spacy.get_nlp(doc_id)

                for entity_id in set([ent.ent_id_ for ent in nlp_doc.ents]):
                    short, long = self.abbrev_df.loc[int(entity_id)][['short', 'long']]
                    entity_occurrences = [ent for ent in nlp_doc.ents if ent.ent_id_ == entity_id]
                    d1_parts = []
                    d2_parts = []
                    current_pos = 0
                    for oc in entity_occurrences:
                        d1_parts.append(nlp_doc[current_pos:oc.start].text_with_ws)
                        d2_parts.append(nlp_doc[current_pos:oc.start].text_with_ws)

                        d1_parts.append(short)
                        d2_parts.append(long)

                        if oc.text != oc.text_with_ws:
                            d1_parts.append(' ')
                            d2_parts.append(' ')

                        current_pos = oc.end
                    d1_parts.append(nlp_doc[current_pos:].text_with_ws)
                    d2_parts.append(nlp_doc[current_pos:].text_with_ws)
                    
                    yield ProbeSample(qid=r.query_id, doc_a_id=None, doc_b_id=None,
                                        query_text=nlp_query.text, 
                                        doc_a_text=''.join(d1_parts), 
                                        doc_b_text=''.join(d2_parts),
                                        metadata={
                                            'short': short,
                                            'long': long,
                                            'short_in_query': any([ent.ent_id_ == entity_id and ent.text.lower() == short.lower() for ent in nlp_query.ents]),
                                            'long_in_query': any([ent.ent_id_ == entity_id and ent.text.lower() == long.lower() for ent in nlp_query.ents]),
                                        })
