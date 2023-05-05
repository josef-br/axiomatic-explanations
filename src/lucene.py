import lucene
if lucene.getVMEnv() is None:
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

from lucene import collections

from .util import stopwords
from pathlib import Path
from typing import List, Generator, Tuple
from .dataset import DocumentSet, QuerySet, Ranking, RankingSet
import logging
from itertools import starmap, chain
from more_itertools import ichunked
from tqdm.auto import tqdm
import pandas as pd
import math
import numpy as np

from org.apache.lucene.analysis import LowerCaseFilter, StopFilter, CharArraySet, Analyzer
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.analysis.standard import StandardTokenizer
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute
from org.apache.lucene.analysis.en import PorterStemFilter, EnglishAnalyzer
from org.apache.lucene.analysis.de import GermanStemFilter
from java.nio.file import Paths
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, Term, IndexOptions, FieldInvertState
from org.apache.lucene.document import Document, Field, StringField, TextField, FieldType
from org.apache.lucene.store import NIOFSDirectory, FSDirectory
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, TermQuery, BooleanQuery, BooleanClause, ConstantScoreQuery, Sort, TermStatistics, CollectionStatistics, PhraseQuery
from org.apache.lucene.search.similarities import BM25Similarity, TFIDFSimilarity
from org.apache.lucene.util import BytesRefIterator
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper

from java.util import Map


def field_analyzers_german(stopwords_path='resources/stopwords.txt'):
    stopwords_list = stopwords(stopwords_path)
    return {
        'text': CustomAnalyzer(stem=True, lang='de', stopwords=[]),
        'text_stopped': CustomAnalyzer(stem=True, lang='de', stopwords=stopwords_list),
        'text_raw': CustomAnalyzer(stem=False, lang='de', stopwords=[]),
        'text_stopped_raw': CustomAnalyzer(stem=False, lang='de', stopwords=stopwords_list)
    }

def field_analyzers_english(stopwords_path='resources/inquery'):
    stopwords_list = stopwords(stopwords_path)
    return {
        'text': CustomAnalyzer(stem=True, lang='en', stopwords=[]),
        'text_stopped': CustomAnalyzer(stem=True, lang='en', stopwords=stopwords_list),
        'text_raw': CustomAnalyzer(stem=False, lang='en', stopwords=[]),
        'text_stopped_raw': CustomAnalyzer(stem=False, lang='en', stopwords=stopwords_list)
    }

class CustomAnalyzer(PythonAnalyzer):
    def __init__(self, stem=True, lang='en', stopwords=[]):
        super().__init__(Analyzer.PER_FIELD_REUSE_STRATEGY)
        assert lang in ['en', 'de'], f"Invalid language '{lang}'. Use either 'en' or 'de'."
        self.lang = lang
        self.stem = stem
        self.stopwords = stopwords


    def createComponents(self, fieldName:str):
        # always tokenize and lowercase
        source = StandardTokenizer()
        filter = LowerCaseFilter(source)

        # sometimes stop
        if self.stopwords:
            filter = StopFilter(filter, CharArraySet(collections.JavaList(self.stopwords), True))

        # sometimes stem based on language
        if self.stem:
            if self.lang == 'en':
                filter = PorterStemFilter(filter)
            elif self.lang == 'de':
                filter = GermanStemFilter(filter)

        return self.TokenStreamComponents(source, filter)

    def initReader(self, fieldName, reader):
        return reader

    def analyze(self, text, field_name='text') -> List[str]:
        ts = self.tokenStream(field_name, text)
        term_attr = ts.addAttribute(CharTermAttribute.class_)
        ts.reset()
        toks = []
        while(ts.incrementToken()):#
            toks.append(term_attr.toString())
        ts.close()
        return toks



class LuceneIndex:
    def __init__(self, index_path, lang='en'):
        assert lang in ['en', 'de'], f"Invalid language '{lang}'. Use either 'en' or 'de'."
        self.index_dir = FSDirectory.open(Paths.get(str(index_path)))
        self.analyzers = field_analyzers_english() if lang == 'en' else field_analyzers_german()

        self.reader = DirectoryReader.open(self.index_dir)
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(BM25Similarity())

    def __del__(self):
        self.reader.close()
        self.index_dir.close()

    def _resolve_internal_id(self, external_id):
        query = TermQuery(Term('id', str(external_id)))
        res = self.searcher.search(query, 1)
        if res.totalHits.value == 1:
            return res.scoreDocs[0].doc
        else:
            raise FileNotFoundError(f"Could not find a document with id {external_id}")

    def _resolve_external_id(self, internal_id):
        return self.reader.document(internal_id).get('id')

    def _pseudo_parser(self, text, field_name):
        """Lucene queryparser handles operators and this fancy shit... We don't need that, we just want to match the tokens.
        --> Run the query though the same analyzer as indexing, build a Boolean-Should-Query of terms"""
        builder = BooleanQuery.Builder()
        for term in self.tokenize(text, field_name):
            builder.add(TermQuery(Term(field_name, term)), BooleanClause.Occur.SHOULD)
        return builder.build()

    def search_bm25(self, query_text, k=10, field_name='text_stopped', query_id=None):
        query = self._pseudo_parser(query_text, field_name)
        search_result = self.searcher.search(query, k)
        return Ranking(query_id=query_id, dataframe=pd.DataFrame.from_records([
            {'qid': query_id,
            'rank': idx+1,
            'result_id': self._resolve_external_id(r.doc),
            'score': r.score}
            for idx, r in enumerate(search_result.scoreDocs)
        ], columns=['qid', 'rank', 'result_id', 'score']))

    def search_bm25_batch(self, queries: QuerySet, k=10, show_progress_bar=True, field_name='text_stopped'):
        return RankingSet.accumulate(tqdm(starmap(lambda id, text: self.search_bm25(text, k, field_name, id), 
                                                queries.iter_queries()), 
                                        total=queries.num_queries()))

    def tokenize(self, text, field_name='text') -> List[str]:
        return self.analyzers[field_name].analyze(text, field_name)
    
    def tfs(self, docid, query_terms=None, field_name='text'):
        internal_id = self._resolve_internal_id(docid)

        term_vector = self.reader.getTermVector(internal_id, field_name)
        tfs = {}
        if term_vector: 
            term_iterator = term_vector.iterator()
            for term in BytesRefIterator.cast_(term_iterator):
                postings_iterator = term_iterator.postings(None)
                postings_iterator.nextDoc()  # prime the enum which works only for the current doc
                tfs[term.utf8ToString()] = postings_iterator.freq()

        if query_terms is None:
            return tfs
        else:
            return {term: tfs.get(term, 0) for term in query_terms}

    def positions(self, docid, query_terms=None, field_name='text'):
        internal_id = self._resolve_internal_id(docid)

        term_vector = self.reader.getTermVector(internal_id, field_name)
        tfs = {}
        if term_vector: 
            term_iterator = term_vector.iterator()
            for term in BytesRefIterator.cast_(term_iterator):
                postings_iterator = term_iterator.postings(None)
                postings_iterator.nextDoc()  # prime the enum which works only for the current doc
                tfs[term.utf8ToString()] = [postings_iterator.nextPosition() for _ in range(postings_iterator.freq())]

        if query_terms is None:
            return tfs
        else:
            return {term: tfs.get(term, []) for term in query_terms}

    def idf(self, term, field_name='text'):
        doc_num = self.reader.getDocCount(field_name)
        doc_freq = self.reader.docFreq(Term(field_name, term))
        return math.log((doc_num - doc_freq + .5) / (doc_freq + .5) + 1.)
    
    def term_count(self, term, field_name='text'):
        """How often is this term contained in the corpus"""
        return self.reader.docFreq(Term(field_name, str(term)))

    def doclen(self, docid, field_name='text'):
        internal_id = self._resolve_internal_id(docid)
        term_vector = self.reader.getTermVector(internal_id, field_name)
        return term_vector.getSumTotalTermFreq() if term_vector else 0

    def score_bm25_known(self, query_text, doc_id, field_name='text'):
        query = self._pseudo_parser(query_text, field_name)
        return self.searcher.explain(query, self._resolve_internal_id(doc_id)).getValue().floatValue()

    def get_doc_text(self, docid) -> str:
        internal_id = self._resolve_internal_id(docid)
        return self.reader.document(internal_id).get('text')

    def sample_documents_containing(self, text: str, field_name='text_raw', sample_size=32) -> Generator[Tuple[str, str], None, None]:
        """We need this for probing: Find/iter some documents that contain a certain phrase"""
        toks = self.tokenize(text, field_name)
        
        # TODO If we wanted to sample random documents we would have to write a custom way to iterate and collect search results.
        # In a nutshell: Write a Collector and LeafCollector that collects only the wanted documents: https://lucene.apache.org/core/9_4_1/core/org/apache/lucene/search/LeafCollector.html
        # But overriding Java Classes in this Java-C++-Python-Abomination is not that easy: https://lucene.apache.org/pylucene/jcc/features.html
        # so: Just take the top n Documents

        query = ConstantScoreQuery(PhraseQuery(field_name, toks))
        documents = self.searcher.search(query, sample_size, Sort.INDEXORDER, False)
        for doc in documents.scoreDocs:
            internal_doc = self.reader.document(doc.doc)
            yield internal_doc.get('id'), internal_doc.get('text')

    def score_bm25_unknown(self, query_text, document_text, field_name='text'):
        """Unlike terrier, unfortunately, whoosh does not offer an API to score an arbitrary text against an arbitrary document,
        it does only do indexed documents. But we need this functionality for Text Manipulation Probes.
        Sooo... Imitate the internal processing of indexing and scoring"""
        doc_tokenized = self.tokenize(document_text, field_name)
        query_tokenized = self.tokenize(query_text, field_name)
        if len(doc_tokenized) == 0 or len(query_tokenized) == 0: return .0

        doc_tfs = { term: doc_tokenized.count(term) for term in set(doc_tokenized)}

        # do what lucene would do when indexing the document
        doc_index_infos = FieldInvertState(0, field_name, IndexOptions.DOCS_AND_FREQS, len(doc_tokenized)-1, len(doc_tokenized), 0, len(doc_tokenized[-1])-1, max(doc_tfs.values()), len(doc_tfs))
        similarity = BM25Similarity()
        field_norm = similarity.computeNorm(doc_index_infos)
        coll_stats = CollectionStatistics(field_name, max(self.reader.maxDoc(), 1), max(self.reader.getDocCount(field_name), 1), max(self.reader.getSumTotalTermFreq(field_name), len(doc_tfs)), max(self.reader.getSumDocFreq(field_name), len(doc_tokenized)))
        def term_stats(term):
            l_term = Term(field_name, term)
            doc_freq = self.reader.docFreq(l_term) + int(term in doc_tfs)
            if doc_freq > 0:
                return TermStatistics(l_term.bytes(), doc_freq, self.reader.totalTermFreq(l_term) + doc_tfs.get(term, 0))
            else:
                return None

        q_term_stats = {term: term_stats(term) for term in set(query_tokenized)}

        return sum([similarity.scorer(1., coll_stats, q_term_stats[term]).score(float(doc_tfs.get(term, 0)), field_norm) for term in query_tokenized if q_term_stats[term]])

    @staticmethod
    def build(docset: DocumentSet,
            index_path: Path,
            index_name: str,
            lang='en',
            stopwords_path=None,
            batch_size=None,
            show_progress_bar=True):
        assert lang in ['en', 'de'], f"Invalid language '{lang}'. Use either 'en' or 'de'."
        if stopwords_path is None:
            if lang == 'en':
                stopwords_path = Path('resources/inquery')
        
        path = Path(index_path) / index_name
        path.mkdir(parents=True)

        logging.info(f"Building Lucene Index {index_name} at {str(index_path)}")

        # This defines the fields: 'text' is stored, the others are not (beacuse it is always the same raw content)
        text_type = FieldType()
        text_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
        text_type.setStoreTermVectors(True)
        text_type.setStoreTermVectorPositions(True)

        stored_text_type = FieldType(text_type)
        stored_text_type.setStored(True)
        # The text is copied into three fields. The Analyzer class defines how each is analyzed (determined by their name)
        def create_lucene_document(id, text):
            lucene_doc = Document()
            lucene_doc.add(Field("id", str(id), StringField.TYPE_STORED))
            lucene_doc.add(Field("text", str(text), stored_text_type))
            lucene_doc.add(Field('text_stopped', str(text), text_type))
            lucene_doc.add(Field('text_raw', str(text), text_type))
            return lucene_doc

        analyzers = field_analyzers_english() if lang == 'en' else field_analyzers_german()
        analyzerWrapper = PerFieldAnalyzerWrapper(
            CustomAnalyzer(stem=False, lang=lang, stopwords=[]), # default
            Map.of(*list(chain(*analyzers.items())))
        )

        config = IndexWriterConfig(analyzerWrapper)

        index_dir = FSDirectory.open(Paths.get(str(path)))
        writer = IndexWriter(index_dir, config)
        try:
            document_iterator = starmap(create_lucene_document, docset.iter_for_indexing())

            if batch_size is None: batch_size = docset.num_documents()
            for doc_batch in ichunked(tqdm(document_iterator, total=docset.num_documents(), disable=not show_progress_bar), batch_size):
                for doc in doc_batch:
                    writer.addDocument(doc)

                writer.commit()

            logging.info(f"Successfully indexed {docset.num_documents()} documents.")
        finally:
            writer.close()
            index_dir.close()

        return LuceneIndex(path, lang) 
