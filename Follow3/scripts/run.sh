mkdir -p ../Results
pip install -r requirements.txt 
python Initial_Snorkel_Pipeline.py ../../Datasets/NEW_FNC/2020-11.csv ../Results/NEW_FNC_2020-11.txt ../../Datasets/NEW_FNC/2020-11-snorkel.csv
python Initial_Snorkel_Pipeline.py ../../Datasets/NEW_FNC/2020-12.csv ../Results/NEW_FNC_2020-12.txt ../../Datasets/NEW_FNC/2020-12-snorkel.csv