dirs=(
#data
#../data/train.parquet
gcn_morgan-1024
split1_nn_morgan-1024
split1_separate-train_nn_morgan-1024
split2_nn_morgan
random_split_nn_morgan-1024
conda*
scripts)

for dir in ${dirs[@]}; do
  rsync -rltvha $dir mynguyen@discovery.usc.edu:/scratch1/mynguyen/leash/

done
#rsync -rltvha --exclude 'log*.out' --exclude 'icm/' --exclude '*checkpoint*.pth' --exclude 'train.pkl'  --exclude 'nn*/*pkl' --exclude '*train.csv' --exclude '*train.parquet'  mynguyen@hpc-transfer1.usc.edu:/scratch1/mynguyen/leash-BELKA/ .
