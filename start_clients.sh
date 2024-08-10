
NUM_CLIENTS=$2
dataset_dir=$1

cd  "brain_tumor_detection"
for ((i=1; i<=NUM_CLIENTS; i++))
do
    echo "Istanza del client $i avviata"
    python client.py "$dataset_dir" "$i" &
done