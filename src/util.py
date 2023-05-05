"""Some useful functions that are too long and ugly to be kept in notebooks"""

from pathlib import Path
from typing import List
import re
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
from functools import reduce
import importlib
from itertools import starmap, chain, permutations
from .get_data import Data
from .diagnostic import DiagnosticDataset
from DST.word_discrimination import WordDiscrimination

# A pattern that matches either a single non-whitespace char (\S) or a sequence of alphanumeric chars.
# The sequence can contain the special chars -'&, but only then they are not at the beginning or end of the sequence.
# Thereby <'word> gets split into two (<'>, <word>), but <don't> stays together as one token
TOKEN_PATTERN_ONLY_ALPHANUM = r"(?:(?<!^)(?<!\W)[\-\'&\.](?!\W)(?!$)|[A-Za-z0-9])+"
TOKEN_PATTERN = fr"{TOKEN_PATTERN_ONLY_ALPHANUM}|\S"

"""Access custom stopword lists"""
def _read_file_lines(file_path: Path) -> List[str]:
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File {str(file_path)} does not exist or is a dir")
    with open(file_path) as file:
        lines = [line.rstrip() for line in file]

    return lines

def stopwords(file_path=Path('resources/inquery')):
    return _read_file_lines(Path(file_path))

def stopphrases(file_path=Path('resources/stopStructure')):
    return _read_file_lines(Path(file_path))

"""Transform Fulltext Queries to Keyword queries"""
def reduce_to_keyword_queries(queries: pd.DataFrame, q_text_fieldname='text', stopphrase_list=None, stopword_list=None, max_len=5):
    stopphrase_list = stopphrase_list or stopphrases()
    stopword_list = stopword_list or stopwords()

    stopphrase_list = [phrase + ' ' for phrase in stopphrase_list] # do not accidently cut a word in half
    token_pattern = re.compile(TOKEN_PATTERN_ONLY_ALPHANUM)

    processing_steps = [
        lambda q: re.split(r'(\.|\?|\!)\s', q)[0].strip(),                                                  # step 0: if multiple sentences then keep only first one
        lambda q: ' '.join([str(t).lower() for t in token_pattern.findall(q)]),                             # step 1: Clean: keep only alphanum tokens and convert to lowercase
        lambda q: q.replace(next((stop for stop in stopphrase_list if q.startswith(stop)), ''), '', 1),     # step 2: stop structure: remove start of string if it is a stop structure
        lambda q: remove_stopwords(str(q), stopword_list),                                                  # step 3: stopwords: remove stopwords
    ]
    processing_pipeline = reduce(lambda pipe, step: lambda input: step(pipe(input)), processing_steps)

    return queries.assign(**{q_text_fieldname: lambda df: df[q_text_fieldname].apply(processing_pipeline)}) \
                    .assign(len=lambda df: df[q_text_fieldname].apply(lambda q: len(token_pattern.findall(q)))) \
                    .query(f'len <= {max_len}')


def create_results_dfs(dataobj: Data, 
                            name=None,
                            datasets = {
                                'booltf': None,
                                'tf': None,
                                'idf': None,
                                'len': None,
                                'proximity': None,
                                'ordering': None,
                                'exact_match': None,
                                'lemmatize': None,
                                'stopwords': None,
                                'spellerror': None,
                                'succinctness': None,
                                'thesaurus': {
                                    '_none': '~short_in_query and ~long_in_query',
                                    '_qlong': '~short_in_query and long_in_query',
                                    '_qshort': 'short_in_query and ~long_in_query',
                                }
                            },
                            querysets={'keyword': 'keyword', 'verbose': 'verbose'}):
    if name is None: name = dataobj.path.name
    collected_results = [
        {
            'queryset': querysets[qset_name],
            'diagnostic': f"{scored_dataset.diag_data.probe_name.replace('.csv', '')}{postfix}",
            'model': model,
            'kendall': scored_dataset.kendall_tau_b(),
            'size': scored_dataset.diag_data.df.shape[0],
            'is_significant': int(scored_dataset.is_significant())
        }
        for qset_name in dataobj.list_probe_qsets()
        for model in ['bm25', 'neural']
        for postfix, scored_dataset in chain.from_iterable(starmap(
                    lambda pname, queries: 
                        [('', DiagnosticDataset(dataobj, qset_name, pname).scored(model))] if queries is None 
                        else [(postfix, DiagnosticDataset(dataobj, qset_name, pname).scored(model).query(query))
                              for postfix, query in queries.items()], 
                    datasets.items()))
        if qset_name in querysets.keys()
    ]
    # for correct sorting:
    dataset_order = list(chain.from_iterable([[pname] if queries is None 
                                              else [f'{pname}{postfix}' for postfix in queries.keys()]
                                              for pname, queries in datasets.items()]))

    correlations_df = pd.DataFrame(collected_results).pivot(columns=['model'], index=['diagnostic', 'queryset'], values='kendall') \
        .loc[dataset_order] \
        .round(3)
    
    significant_df = pd.DataFrame(collected_results).pivot(columns=['model'], index=['diagnostic', 'queryset'], values='is_significant') \
            .loc[dataset_order] \
            .round(3)
    
    return correlations_df, significant_df

                
def build_abbreviation_dataframe_from_wordnet():
    wn = importlib.import_module('nltk.corpus').wordnet
    abbrevs = [
        (short.replace('_', ' '), long.replace('_', ' '))
        for synset in wn.all_synsets('n')
        for short, long in permutations(map(lambda t: t.lower(), synset.lemma_names()), 2)
        if WordDiscrimination.default_classify_func(long, short) == 'abbreviation'
    ]
    return pd.DataFrame.from_records(abbrevs, columns=['short', 'long']).sort_values(by='short').reset_index(drop=True)


def abbreviation_df_to_spacy_patterns(abbrev_df, spacy_nlp):
    return list(chain.from_iterable([[{
                'label': 'MINE',
                'id': str(t.Index),
                'pattern': [{'LOWER': term.lower_ } for term in spacy_nlp.tokenizer(t.short)]
            }, {
                'label': 'MINE',
                'id': str(t.Index),
                'pattern': [{'LOWER': term.lower_ } for term in spacy_nlp.tokenizer(t.long)]
            }]
        for t in abbrev_df.itertuples()
    ]))