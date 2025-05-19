# _except_mvtec_visa
# _except_continual_ad

json_path_list=(
    "5classes_tasks"
    "5classes_tasks_except_mvtec_visa"
)


for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=12; continual_model_id>=1; continual_model_id--)); do
        python test.py \
            --img_size 336 \
            --batch_size 16 \
            --device cuda \
            --json_path $json_path \
            --task_id 0 \
            --continual_model_id $continual_model_id
    done
done


json_path_list=(
    "5classes_tasks_except_continual_ad"
)


for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=6; continual_model_id>=1; continual_model_id--)); do
        python test.py \
            --img_size 336 \
            --batch_size 16 \
            --device cuda \
            --json_path $json_path \
            --task_id 0 \
            --continual_model_id $continual_model_id
    done
done




json_path_list=(
    "10classes_tasks"
    "10classes_tasks_except_mvtec_visa"
)


for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=6; continual_model_id>=1; continual_model_id--)); do
        python test.py \
            --img_size 336 \
            --batch_size 16 \
            --device cuda \
            --json_path $json_path \
            --task_id 0 \
            --continual_model_id $continual_model_id
    done
done


json_path_list=(
    "10classes_tasks_except_continual_ad"
)


for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=3; continual_model_id>=1; continual_model_id--)); do
        python test.py \
            --img_size 336 \
            --batch_size 16 \
            --device cuda \
            --json_path $json_path \
            --task_id 0 \
            --continual_model_id $continual_model_id
    done
done


json_path_list=(
    "30classes_tasks"
    "30classes_tasks_except_mvtec_visa"
)


for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=2; continual_model_id>=1; continual_model_id--)); do
        python test.py \
            --img_size 336 \
            --batch_size 16 \
            --device cuda \
            --json_path $json_path \
            --task_id 0 \
            --continual_model_id $continual_model_id
    done
done


json_path_list=(
    "30classes_tasks_except_continual_ad"
)

for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    for ((continual_model_id=1; continual_model_id>=1; continual_model_id--)); do
        python test.py \
            --img_size 336 \
            --batch_size 16 \
            --device cuda \
            --json_path $json_path \
            --task_id 0 \
            --continual_model_id $continual_model_id
    done
done

