"""Wrapper for used SpaCy functionality"""

import spacy
from pathlib import Path
import mmap
from typing import Tuple, Union, Iterable
import pandas as pd
from more_itertools import ichunked
from tqdm.autonotebook import tqdm
import json
from itertools import tee


class SpacyIndex:
    """Spacy processes a text and extracts semantic information. At large scale this takes some time...
    To avoid doing this over and over and over, this class provides the opportunity to batch process an 
    entire corpus (either a documentset or a queryset) and access the processed items per id.
    The spacy data can get quite large... so this a proprietary implementation with memory mapping.
    """
    def __init__(self, path: Path):
        if not path.stem == 'spacy' and 'spacy' in path.iterdir():
            path = path / 'spacy'
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Path {str(path)} is invalid.")

        with open(path / 'config.json', 'r') as file:
            self.config = json.load(file)
        
        if not spacy.util.is_package(self.config['spacy_model_name']) and not Path(self.config['spacy_model_name']).exists():
            raise ModuleNotFoundError(f"Spacy model {self.config['spacy_model_name']} is not installed")
        self.nlp = spacy.load(self.config['spacy_model_name'])

        self.index_df = pd.read_csv(path / 'index.csv', index_col='id')
        self.data_path = path / 'data.nlp'

    def get_nlp(self, docid) -> spacy.tokens.Doc:
        if docid not in self.index_df.index: 
            raise ValueError(f"Doc-ID {docid} not found in index.")
        pos_info = self.index_df.loc[docid]
        with open(self.data_path, 'r+b') as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                bytes = mmap_obj[pos_info.start_offset:pos_info.start_offset+pos_info.byte_size]
                return spacy.tokens.Doc(self.nlp.vocab).from_bytes(bytes, exclude=self.config['exclude_fields'])

    def process_text(self, text: str) -> spacy.tokens.Doc:
        return self.nlp(text)

    @staticmethod
    def build(path: Path, 
            docs_with_ids: Iterable[Tuple[Union[str, int], str]],
            size: int, 
            spacy_model_name: str, 
            show_progress_bar=True,
            spacy_batch_size=1000,
            processes=1,
            force_cpu=False,
            spacy_exclude=['sentiment', 'tensor', 'user_data']):
        s_index_path = path / 'spacy'
        s_index_path.mkdir(parents=True, exist_ok=False)

        if not spacy.util.is_package(spacy_model_name) and not Path(spacy_model_name).exists():
            raise ModuleNotFoundError(f"Spacy model {spacy_model_name} is not installed")

        if force_cpu:
            spacy.require_cpu()
        else:
            spacy.prefer_gpu()
        nlp = spacy.load(spacy_model_name)

        # write model name to config for later use
        config = { 'spacy_model_name': str(spacy_model_name), 'exclude_fields': spacy_exclude }
        with open(s_index_path / 'config.json', 'w') as file:
            json.dump(config, file)

        index_file_path = s_index_path / 'index.csv'
        data_file_path = s_index_path / 'data.nlp'        
        
        index = []
        start_offs = 0

        # bit complicated but worth it:
        # we have one iterator containing (id, text)
        # -> duplicte it into two iterators, make one return only the id and the other return the processed text
        # this allows to use the awesome spacy built-in for batch processing (about 3x faster)
        it0, it1 = tee(docs_with_ids, 2)
        iterator = zip(
            map(lambda doc: doc[0], it0), # get the ID from the <id, text> tuple
            nlp.pipe(map(lambda doc: str(doc[1]), it1), batch_size=spacy_batch_size, n_process=processes) # get the text from the tuple and run it through the spacy batch processer
        )
        with open(data_file_path, 'ab') as file:
            for id, text_nlp in tqdm(iterator, total=size, disable=not show_progress_bar):
                processed_bin = text_nlp.to_bytes(exclude=spacy_exclude)
                index.append((id, start_offs, len(processed_bin)))
                start_offs += len(processed_bin)
                file.write(processed_bin)

        pd.DataFrame.from_records(index, columns=['id', 'start_offset', 'byte_size']).to_csv(index_file_path, index=False)
