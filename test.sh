python test.py \
    --phase base \
    --img_size 336 \
    --batch_size 16 \
    --base_json base_classes \
    --task_json 5classes_tasks \
    --device cuda \
    --pretrained_path outputs/HGAD_base_img.pt \
    --scores_dir scores