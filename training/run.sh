# !/bin/bash
# an example for ASP fine-tuning
# fix the seed to 2048 in order to reproduce the result

for model_name in 'deberta-v3-large' 'deberta-v3-base' 'bert-large-cased' 'bert-base-cased'
do
    for D in 'length' 'mass' 'price' 'temperature'
    do
        CUDA_VISIBLE_DEVICES="5,6,7,0" python -u ASP_training.py \
        --task_name $D \
        --model_type $model_name \
        --seed 2048 \

    done
done
