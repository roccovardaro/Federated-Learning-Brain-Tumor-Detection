# shellcheck disable=SC2164
cd 'brain_tumor_detection'

dataset_dir=$1
python client.py "$dataset_dir"
