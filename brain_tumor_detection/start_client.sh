# shellcheck disable=SC2164
numero_casuale=$(awk "BEGIN {print $RANDOM/32767}")

dataset_dir1="datasets/brain_tumor_dataset"
dataset_dir2="datasets/brain_tumor_dataset2"


# Controlla se il numero è maggiore o uguale a 0.5
if (( $(echo "$numero_casuale >= 0.5" | bc -l) )); then
    echo "Il numero casuale è maggiore o uguale a 0.5"
    python client.py "$dataset_dir1"
else
    echo "Il numero casuale è minore di 0.5"
     python client.py "$dataset_dir2"
fi

