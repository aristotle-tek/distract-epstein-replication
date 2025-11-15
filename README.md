# Replication data: Distracting from the Epstein files 

Repository code for "Distracting from the Epstein files? Attention and short-run shifts in Trump's Truth Social posts"


## Package versions

Python 3.11+ recommended.

Minimal deps: `click`, `numpy`, `pandas`, `pyarrow`,`matplotlib`, `sentence-transformers`, `statsmodels`, `matplotlib`, `pyyaml`.
Or use:
```bash
pip install -r requirements.txt
````


## General notes
- use UTC for all timestamps
- time window:  2024-10-01  → 2025-10-19
- `config.yml` - for paths


## Data
- Trump truth social posts comes from the [Trump Truth Social Archive (CC0 1.0 Universal)](https://github.com/stiles/trump-truth-social-archive)
- TV transcripts come from Internet Archive. These are extensive so we provide just a small sample, and then provide the processed data files such as the embeddings.
- To include the embeddings files, I split them into 6 files so that users can download the repository without using git lfs. They need to be re-constructed by running `python src/data_prep/merge_embeddings.py`

## Analysis 
- All analyses can be re-run with `bash scripts/_run_all.sh` (after reconstructing the embeddings file)
- Or run subsets with the other scripts, or run directly with python.





