echo "========start to test bert1============="
python ../Models/BERT/eval.py ../Save_Models/bert1 ../../Datasets/NEW_FNC/2020-11.csv ../Results/BERT/bert1_NEW_FNC_2020-11.txt
python ../Models/BERT/eval.py ../Save_Models/bert1 ../../Datasets/NEW_FNC/2020-12.csv ../Results/BERT/bert1_NEW_FNC_2020-12.txt

echo "========start to test bert2============="
python ../Models/BERT/eval.py ../Save_Models/bert2 ../../Datasets/NEW_FNC/2020-11.csv ../Results/BERT/bert2_NEW_FNC_2020-11.txt
python ../Models/BERT/eval.py ../Save_Models/bert2 ../../Datasets/NEW_FNC/2020-12.csv ../Results/BERT/bert2_NEW_FNC_2020-12.txt

echo "========start to test bert3============="
python ../Models/BERT/eval.py ../Save_Models/bert3 ../../Datasets/NEW_FNC/2020-11.csv ../Results/BERT/bert3_NEW_FNC_2020-11.txt
python ../Models/BERT/eval.py ../Save_Models/bert3 ../../Datasets/NEW_FNC/2020-12.csv ../Results/BERT/bert3_NEW_FNC_2020-12.txt

echo "========start to test bert4============="
python ../Models/BERT/eval.py ../Save_Models/bert4 ../../Datasets/NEW_FNC/2020-11.csv ../Results/BERT/bert4_NEW_FNC_2020-11.txt
python ../Models/BERT/eval.py ../Save_Models/bert4 ../../Datasets/NEW_FNC/2020-12.csv ../Results/BERT/bert4_NEW_FNC_2020-12.txt