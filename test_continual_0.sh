# _except_mvtec_visa
# _except_continual_ad



for ((continual_model_id=12; continual_model_id>=1; continual_model_id--)); do
    for ((task_id=1; task_id<=continual_model_id; task_id++)); do
        echo "Running continual_model_id=$continual_model_id, task_id=$task_id"
        python test.py \
            --img_size 336 \
            --batch_size 64 \
            --device cuda \
            --json_path 5classes_tasks_except_mvtec_visa \
            --task_id $task_id \
            --continual_model_id $continual_model_id
    done
done