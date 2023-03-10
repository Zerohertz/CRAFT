read -p "Exp: " dir
rm -rf ./exp/$dir
CUDA_VISIBLE_DEVICES=1 python test.py --yaml=main-org --dir=$dir
cp ./config/main.yaml ./exp/$dir/