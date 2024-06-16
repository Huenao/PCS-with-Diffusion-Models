#!/bin/bash

# 定义字典
declare -A subject_with_cls=(
  ["backpack"]="backpack"
  ["backpack_dog"]="backpack"
  ["bear_plushie"]="stuffed animal"
  ["berry_bowl"]="bowl"
  ["can"]="can"
  ["candle"]="candle"
  ["cat"]="cat"
  ["cat2"]="cat"
  ["clock"]="clock"
  ["colorful_sneaker"]="sneaker"
  ["dog"]="dog"
  ["dog2"]="dog"
  ["dog3"]="dog"
  ["dog5"]="dog"
  ["dog6"]="dog"
  ["dog7"]="dog"
  ["dog8"]="dog"
  ["duck_toy"]="toy"
  ["fancy_boot"]="boot"
  ["grey_sloth_plushie"]="stuffed animal"
  ["monster_toy"]="toy"
  ["pink_sunglasses"]="glasses"
  ["poop_emoji"]="toy"
  ["rc_car"]="toy"
  ["red_cartoon"]="cartoon"
  ["robot_toy"]="toy"
  ["shiny_sneaker"]="sneaker"
  ["teapot"]="teapot"
  ["vase"]="vase"
  ["wolf_plushie"]="stuffed animal"
  ["elephant"]="statue"
  ["thin_bird"]="statue"
  ["physics_mug"]="mug"
  ["clock2"]="clock"
  ["colorful_teapot"]="teapot"
  ["round_bird"]="statue"
  ["red_teapot"]="teapot"
  ["cat_statue"]="statue"
  ["mug_skulls"]="statue"
)

# 模型名称
MODEL_NAME="runwayml/stable-diffusion-v1-5"

# 数据集根目录
DATASET_DIR="../../pcs_dataset/subjects"

# 遍历字典中的所有键值对
for subject in "${!subject_with_cls[@]}"; do
  INIT_TOKEN=${subject_with_cls[$subject]}
  
  # 设置数据目录
  DATA_DIR="$DATASET_DIR/$subject"
  
  # 运行 accelerate launch 命令
  accelerate launch --main_process_port=29517 \
    --gpu_ids=4 \
    textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATA_DIR \
    --learnable_property="object" \
    --placeholder_token="<new1>" \
    --initializer_token=$INIT_TOKEN \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=3000 \
    --learning_rate=5.0e-04 \
    --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir="../../logs/subjects/textual_inversion/$subject"
done
