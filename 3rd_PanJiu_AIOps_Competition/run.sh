rm -rf model
#unzip model.zip
python3 code/get_crashdump_venus_fea.py
python3 code/catboost_fs.py
zip -j result.zip prediction_result/catboost_result.csv
