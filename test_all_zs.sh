json_path_list=(
    "meta_continual_ad_test_total"
)

scenario_list=(
    "scenario_3"
)

case_list=(
    "5classes_tasks"
    "10classes_tasks"
    "30classes_tasks"
)

for ((i=0; i<${#json_path_list[@]}; i++)); do
    json_path=${json_path_list[$i]}
    echo "Running for json_path=$json_path"
    
    for ((j=0; j<${#scenario_list[@]}; j++)); do
        scenario=${scenario_list[$j]}
        echo "Scenario=$scenario"
    
        for ((k=0; k<${#case_list[@]}; k++)); do
            case=${case_list[$k]}
            echo "Case=$case"
            
            python test_zs.py \
                --img_size 336 \
                --batch_size 4 \
                --json_path $json_path \
                --scenario $scenario \
                --case $case
        done
    done
done