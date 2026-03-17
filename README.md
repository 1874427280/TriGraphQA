# TriGraphQA
## Installation
1. Download this repository
```bash
git clone https://github.com/1874427280/TriGraphQA.git
```
2. Set up conda environment locally
```bash
cd TriGraphQA
conda env create --name TriGraphQA -f environment.yml
```
3. Activate conda environment
```bash
conda activate TriGraphQA
```
## Usage
Here is the feature_build.py and predict.py script parameters' introduction
```python

python feature_build.py 
--input_dir  Protein complex target
--work_dir  Base directory for intermediate and final outputs
--voronota_bin  Path to voronota binary (The recommended way to get the latest stable version of Voronota and its expansions is to download the latest archive from the "Releases" page:
[https://github.com/kliment-olechnovic/voronota/releases](https://github.com/kliment-olechnovic/voronota/releases).)
--script_dir  ./tool

python predict.py
--graph_dir      Directory containing *_graphs.pt files
--output_dir     Directory to save prediction results
--model_path   Path to the trained .pkl model checkpoint
```
