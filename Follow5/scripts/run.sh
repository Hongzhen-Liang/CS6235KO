# pip install -r requirements.txt 
mkdir -p ../Results/BERT
python Snorkel_Pipeline1.py ../../Datasets/NEW_FNC/2020-12.csv ../Results/BERT/bert5_NEW_FNC_2020-12.txt ../../Datasets/NEW_FNC/2020-12-snorkel1.csv
python Snorkel_Pipeline2.py ../../Datasets/NEW_FNC/2020-12.csv ../Results/BERT/bert6_NEW_FNC_2020-12.txt ../../Datasets/NEW_FNC/2020-12-snorkel2.csv
