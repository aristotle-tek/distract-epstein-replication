# Replication data: Distracting from the Epstein files 

Repository code for ["Distracting from the Epstein files? Attention and short-run shifts in Trump's Truth Social posts"](https://arxiv.org/abs/2511.11532)


## quickstart

You can run the analyses with a few commands:

0. clone the repository
1. Re-constitute embeddings from split with `python src/data_prep/merge_embeddings.py`
2. All analyses run with `bash scripts/_run_all.sh`

(Note this uses the processed versions of the data; we provide only a sample of the transcripts, but contact me if you want the full replication data starting from the transcripts (or use the script to download them from IA)).


## Package versions

Python 3.11+ recommended

Minimal deps: `click`, `numpy`, `pandas`, `pyarrow`,`matplotlib`, `sentence-transformers`, `statsmodels`, `matplotlib`, `pyyaml`.
Or use:
```bash
pip install -r requirements.txt
````


## General notes
- use UTC for all timestamps
- time window:  2024-10-01 to 2025-10-19
- `config.yml` - for paths


## Data
- Trump truth social posts comes from the [Trump Truth Social Archive (CC0 1.0 Universal)](https://github.com/stiles/trump-truth-social-archive)
- TV transcripts come from Internet Archive. These are extensive so we provide just a small sample, and then provide the processed data files such as the embeddings.
- To include the embeddings files, I split them into 6 files so that users can download the repository without using git lfs. They need to be re-constructed by running `python src/data_prep/merge_embeddings.py`

## Analysis 
- All analyses can be re-run with `bash scripts/_run_all.sh` (after reconstructing the embeddings file)
- Or run subsets with the other scripts, or run directly with python.


## tests
You can run a few tests with `pytest`


