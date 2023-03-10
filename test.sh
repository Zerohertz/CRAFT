read -p "Exp: " dir
rm -rf ./exp/$dir
cd utils/cpp_bindings
sh compile.sh
cd ../..

CUDA_VISIBLE_DEVICES=1 python test.py --yaml=org --dir=$dir
# CUDA_VISIBLE_DEVICES=1 python test.py --yaml=main --dir=$dir
cp ./config/main.yaml ./exp/$dir/