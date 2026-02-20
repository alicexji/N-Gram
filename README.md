# N-Gram
 implement a probabilistic language model (N-gram model) that assists with code completion for Java methods

## Overview

This project implements a complete end-to-end pipeline for training and evaluating n-gram language models (**n ∈ {3, 5, 7}**) on Java source code.

The system:

- Mines Java repositories from GitHub
- Extracts and tokenizes Java methods
- Constructs training, validation, and test datasets
- Trains n-gram language models with smoothing
- Evaluates models using perplexity
- Selects the best model via validation performance
- Generates structured JSON prediction outputs

## Python Dependencies
Packages
```
gitpython
javalang
pandas
requests
tqdm
```

## Installation
### Clone the Repository
```
git clone https://github.com/alicexji/N-Gram.git
cd N-Gram
```

### Install Dependencies
Install dependencies using 
```
pip install -r requirements.txt
```

### Pipeline usage
The entire pipeline is controlled through run.py
Available stages:
mine    → Mine repositories and extract methods
split   → Create training/validation/test datasets
train   → Train models and compute validation perplexity
json    → Generate JSON predictions and compute test perplexity
all     → Run complete pipeline

## Running the pipeline
### Mine Java repos and extract methods for training
Fetches the top Java repos from Github and clones them into data/raw/repos
Tokenizes Java methods appyling these preprocessing features:
- Removes non-ASCII characters
- Removes methods with fewer than 10 tokens
- Removes duplicate methods

Additionally, we check that someone has made a commit to the repository within the past year, showing that it is being maintained.

Run:
```
python run.py --stage mine
```

Output will show in data/processed/all_methods.txt

### Split Datasets
Creates:
- T1.txt (≤15,000 methods)
- T2.txt (≤25,000 methods)
- T3.txt (≤35,000 methods)
- val.txt (1,000 methods)
- test_self.txt (1,000 methods)

Run:
```
python run.py --stage split
```

Output will be in the data/processed folder.

### Train models and evaluate validation perplexity
We will train T1, T2, T3 using n = 3, 5, 7

The models will be trained TWICE, once using add-alpha, once using backoff. 
Results of each will be printed out, and we choose the best model (with the lowest perplexity)

Run:
```
python run.py --stage train
```
The attributes of the best model will be stored in results/best_config.json
This step will take a while. Backoff is much more time consuming. Est 1 hour.

### Generate JSON predictions and evaluate test perplexity
The first step should have generated our own test set data/processed/test_self.txt
Insert the professor's provided test set into that same folder data/processed and rename the file to provided.txt

We are going to generate results on both our self created test set and the provided test using the best model that is stored in best_config.json.

Run:
```
python run.py --stage json
```

Output in results folder. You should see two files:
results-self.json
results-provided.json

This step will take a long time because it will most likely choose n = 7 + backoff as the best model. If you only want to check that this step works, you can manually set n = 3 in the best_config.json generated in the previous step.

### Run entire pipeline
Alternatively, you could run the entire pipeline (rather than going in stages) with 
```
python run.py --stage all
```