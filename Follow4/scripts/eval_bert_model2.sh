echo "========start to test bert1============="
python ../../Follow2/Models/BERT/eval.py ../Save_Models/bert1 ../../Datasets/NEW_FNC/2020-12.csv ../Results/BERT/bert1_NEW_FNC_2020-12.txt

echo "========start to test bert2============="
python ../../Follow2/Models/BERT/eval.py ../Save_Models/bert2 ../../Datasets/NEW_FNC/2020-12.csv ../Results/BERT/bert2_NEW_FNC_2020-12.txt