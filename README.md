# LING573
Repository for LING573 coursework.

# Deliverable 4

Command Pipeline:
python evaluate_reddit.py  # to generate humor scores for reddit data
python normalize.py  # to generate normalized upvote scores for reddit data
python run_correlation.py  # to generate output & results

Spearman correlations and plot figures will be output to:
../results/D4/
Outputs of reddit data with predicted humor scores will be sent to:
../outputs/D4/

# Deliverable 3

To Run:
python FFNN.py

To re-extract pretrained embeddings:
python FFNN.py --extract_pretrained

Pretrained embeddings are saved in the 'src/saved' directory.
Pretrained BERT files are here (must be placed in the 'src' directory for use):
https://drive.google.com/drive/folders/1n2QVlrKonIDcNHAD6PlzopQ4EALdcP-z?usp=sharing

The output of our hyperparameter testing is directed to the file locations:
../results/D3_scores.out & ../outputs/D3/{outputfile_name_for_hyperparameter_test}.out

Best model RMSE & hyperparameters will be printed to the bottom of ../results/D3_scores.out


# Deliveralbe 2
To Run:

Install dependencies:
pip install -r requirements.txt

To rerun training:
python main.py --train

To run to and generate output with scores:
python main.py

The output is directed to the console and file locations:
../results/D2_scores.out & ../outputs/D2/output.csv

Due to github;s limitations on file size, we are not able
to host the pretrained files in this repo.

The pretrained files can be found hosted on our google
drive here:
https://drive.google.com/drive/folders/1n2QVlrKonIDcNHAD6PlzopQ4EALdcP-z?usp=sharing

Alternatively, to generate the files yourself, you can rerun the training prcodeure
as outlined above.


