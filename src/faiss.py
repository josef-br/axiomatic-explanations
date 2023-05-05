"""Wrap all needed functionality form the FAISS library"""

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from faiss.contrib.ondisk import merge_ondisk
import logging
from .dataset import DocumentSet, RankingSet, QuerySet
from typing import List, Tuple
from tqdm.autonotebook import tqdm

ID_MAPPING_FILE_NAME = 'id_mapping.csv'

class FaissIndexWrapper:
    def __init__(self, disk_path, 
                search_nprobe=128 # in how many clusters to search. Higher=more accurate, but slower. Can be overwritten in search()
                ):
        self.search_nprobe = search_nprobe
        self.id_mapper = pd.read_csv(Path(disk_path) / ID_MAPPING_FILE_NAME)
        self.index: faiss.Index = faiss.read_index(str(Path(disk_path) / "populated.index"))

        logging.info(f"Loaded faiss index from {str(disk_path)}. Contains {self.index.ntotal} docs.")

    
    def search(self, queries: QuerySet, k=10, nprobe=None) -> RankingSet:
        self.index.nprobe = nprobe if nprobe is not None else self.search_nprobe
        query_ids, query_embeddings = queries.get_all_query_ids_and_embeddings()

        if len(query_embeddings.shape) != 2 or query_embeddings.shape[1] != self.index.d:
            raise ValueError(f"Query Embeddings must match dimension <num_queries> x {self.index.d}")

        distances, neighbors = self.index.search(query_embeddings, k)
        
        int_to_ext_id_mapper = self.id_mapper.set_index('internal_id')
        results = []
        for qindex, qid in enumerate(query_ids):
            for rank in range(neighbors.shape[1]):
                results.append((qid, rank+1, int_to_ext_id_mapper.at[neighbors[qindex, rank], 'external_id'], distances[qindex, rank]))

        ranking_result_dataframe = pd.DataFrame \
            .from_records(results, columns=['qid', 'rank' ,'result_id' , 'score']) \
            .sort_values(by=['qid', 'rank'])
        return RankingSet(df=ranking_result_dataframe)


    def get_document_embedding(self, docid) -> np.ndarray:
        if self.index.direct_map.no():
            self.index.make_direct_map()
        internal_id = int(self.id_mapper.set_index('external_id').loc[docid].internal_id)
        return self.index.reconstruct(internal_id)
        

    @staticmethod
    def build(dataset: DocumentSet, 
            index_path,
            index_name='faiss_index',
            indexing_batch_size = 256, # 2 ** 15 # how many docs to index at once
            faiss_bucket_size = 128, # 2 ** 12 # how many clusters in the index
            docs_id_field='id',
            docs_text_field='text',
            sbert_model_name='distilbert-base-nli-stsb-quora-ranking',
            random_seeed=42, # For randomly drawing initial documents for centroids
            show_progress_bar=True
            ) -> 'FaissIndexWrapper':
        index_dir: Path = Path(index_path) / index_name
        if index_dir.exists():
            raise FileExistsError(f"Path {str(index_dir)} already exists, cannot create an index here.")

        index_dir.mkdir(parents=True)

        # Faiss cannot handle custom IDs, so we have to do the mapping ourselves. We just use a series containing the external IDs
        # and use its consecutive index as internal Faiss IDs. (With reset_index we create a new consecutive index)
        document_ids: pd.Series = dataset.get_all_document_ids().drop_duplicates().reset_index(drop=True)
        # save the id mapping, as we will need it for searching later
        document_ids.reset_index() \
                .rename(columns={'index': 'internal_id', docs_id_field: 'external_id'}) \
                .to_csv(index_dir / ID_MAPPING_FILE_NAME, index=False)

        model: SentenceTransformer = SentenceTransformer(model_name_or_path=sbert_model_name, cache_folder='__model_cache__')

        # A Faiss index is separated into clusters with fixed centroids. A search is only performed in a subset of clusters.
        # We have to tell the index beforehand what the centroids are (--> train the index)
        training_sample_size = min(faiss_bucket_size * 39, dataset.num_documents())
        logging.debug(f"Beginning index training with {training_sample_size} documents")
        # draw and embed random documents for training
        init_training_doc_ids: pd.Series = document_ids.sample(n=training_sample_size, random_state=random_seeed)
        init_training_texts = dataset.get_document_batch_by_ids(ids=init_training_doc_ids.values, fields=[docs_text_field])[docs_text_field].values
        init_embedings: np.ndarray = model.encode(init_training_texts, normalize_embeddings=True)

        trained_index_path = index_dir / 'trained.index'
        index = faiss.index_factory(model.get_sentence_embedding_dimension(), f"IVF{faiss_bucket_size},Flat", faiss.METRIC_INNER_PRODUCT)
        index.train(init_embedings)
        faiss.write_index(index, str(trained_index_path))
        del init_training_texts, init_embedings # save the memory...

        # When there are many documents it becomes unfeasible to hold the entire index in memory while indexing. Therefore we build multiple
        # small indices (one per batch, each based on the same trained index) and merge them afterwards
        logging.debug("Index trained. Start indexing documents now")

        doc_num = document_ids.size
        batches: List[Tuple]  = [(batch_start, min(batch_start + indexing_batch_size, doc_num)) for batch_start in range(0, doc_num, indexing_batch_size)]
        logging.debug(f"Indexing {doc_num} documents: {len(batches)} batches of size {indexing_batch_size}")

        for batch_num, start_stop in tqdm(enumerate(batches), total=len(batches), disable=not show_progress_bar):
            current_batch_ids: pd.Series = document_ids.iloc[start_stop[0]:start_stop[1]]
            current_batch_texts = dataset.get_document_batch_by_ids(ids=current_batch_ids.values, fields=[docs_text_field])[docs_text_field].values
            current_batch_embeddings: np.ndarray = model.encode(current_batch_texts, normalize_embeddings=True, show_progress_bar=show_progress_bar)
            
            index = faiss.read_index(str(trained_index_path))
            index.add_with_ids(current_batch_embeddings, current_batch_ids.index.values)
            faiss.write_index(index, str(index_dir / f"block_{batch_num}.index"))

        logging.debug("Indexed all batches. Merging them now...")
        index = faiss.read_index(str(trained_index_path))
        block_fnames = [str(index_dir / f"block_{batch_num}.index") for batch_num in range(len(batches))]

        merge_ondisk(index, block_fnames, str(index_dir / "merged_index.ivfdata"))
        faiss.write_index(index, str(index_dir / "populated.index"))
        logging.debug("Finished merging. Congrats, all your documents are indexed now in your faiss (pun intended)")

        logging.debug('Cleaning up....')
        for block_file in index_dir.glob('block_*.index'):
            block_file.unlink()

        return FaissIndexWrapper(index_dir, faiss_bucket_size)