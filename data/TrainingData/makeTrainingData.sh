rm -rf data
rm -rf synth
rm -rf images
rm -rf json

# rm -rf synth
# mkdir synth
# cd synth

# for i in \
# 'synthetic_multiline' \
# 'synthetic_dash_numeric'
# do
#     cp -r /home/jovyan/old_nas/3_project_data/TwinReader/twrd-core-jp-dataset/$i/images ./
#     cp -r /home/jovyan/old_nas/3_project_data/TwinReader/twrd-core-jp-dataset/$i/json ./
# done

# cd ..
# rm -rf data
# mkdir data
# cd data

for i in \
'train'
do
    cp -r /home/jovyan/old_nas/3_project_data/TwinReader/twrd-core-jp-dataset/$i/images ./
    cp -r /home/jovyan/old_nas/3_project_data/TwinReader/twrd-core-jp-dataset/$i/json ./
done

python checkData.py