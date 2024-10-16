echo "========start to train bert1============="
python ../Models/BERT/model1.py ../Save_Models/bert1 ../../Datasets/NELA/9-2020.csv

echo "========start to train bert2============="
python ../Models/BERT/model1.py ../Save_Models/bert2 ../../Datasets/NELA/10-2020.csv

echo "========start to train bert3============="
python ../Models/BERT/model1.py ../Save_Models/bert3 ../../Datasets/OLD_FNC/2020-7.csv

echo "========start to train bert4============="
python ../Models/BERT/model1.py ../Save_Models/bert4 ../../Datasets/OLD_FNC/2020-8.csv