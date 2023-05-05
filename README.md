# Axiomatic Explanations

This is the code part of my Master's thesis. It is an attempt to adapt the approach to describe retrieval models through axioms (see [Fang et al.](https://dl.acm.org/doi/pdf/10.1145/1008992.1009004)) to neural ranking models. The basic method is to create *diagnostic datasets*, an approach introduced by [Rennings et al.](https://link.springer.com/content/pdf/10.1007/978-3-030-15712-8_32.pdf), and makes extensive use of the [ABNIRML-Framework](https://github.com/allenai/abnirml) by [MacAvaney et al.](https://doi.org/10.1162/tacl_a_00457).

This implementation uses data from the [CQADupStack](http://nlp.cis.unimelb.edu.au/resources/cqadupstack/) to analyze the model [cqadupstack-msmarco-distilbert-gpl](https://huggingface.co/GPL/cqadupstack-msmarco-distilbert-gpl) and compare it to a standard BM25 model.

## Usage

It is recommended to install the dependencies into a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This code requires the installation of [`pylucene`](https://lucene.apache.org/pylucene/). This one is a but cumbersome, and it cannot be simply installed via one pip dependency. You can either install it using the official installation guide from the documentation, or you can use my (arguably simpler to install) [fork on GitHub]. **Either way, make sure you install it into the virtual environment if you use one.**

## Processing Steps

1. [**Download Data**](./1_download_data.ipynb): Download and filter the CQADupStack-Data, Create a SpaCy-Model
2. [**Setup Resources**](./2_setup_resources.ipynb): Create a document corpus, querysets, search-indices (BM25 and FAISS), spacy-indices, and rankingsets with a document-ranking per query.
3. [**Build Diagnostic Datasets**](./3_build_diagnostic_datasets.ipynb): Iterate through the rankings to create diagnostic datasets that control for different features.
4. [**Evaluate Diagnostic Datasets**](./4_evaluate_diagnostic_datasets.ipynb): Create scores for each diagnostic dataset using both ranking models and display them.
