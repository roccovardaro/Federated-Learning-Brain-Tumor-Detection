
NUM_CLIENTS=$2
dataset_dir=$1
for ((i=1; i<=NUM_CLIENTS; i++))
do
    echo "Istanza del client $i avviata"
    python client.py "$dataset_dir" &
done