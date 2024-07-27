#!/bin/bash

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
  ["elephant"]="animal statue"
  ["thin_bird"]="animal statue"
  ["physics_mug"]="mug"
  ["clock2"]="clock"
  ["colorful_teapot"]="teapot"
  ["round_bird"]="animal statue"
  ["red_teapot"]="teapot"
  ["cat_statue"]="animal statue"
  ["mug_skulls"]="mug"
  ["barn"]="barn"
  ["cat3"]="cat"
  ["teddybear"]="stuffed animal"
  ["tortoise_plushy"]="stuffed animal"
  ["wooden_pot"]="pot"
  ["chair"]="chair"
  ["flower"]="flower"
  ["table"]="table"
)

DATASET_DIR="../../pcs_dataset/subjects/"

for subject in "${!subject_with_cls[@]}"; do

  class=${subject_with_cls[$subject]}
  
  INSTANCE_DIR="$DATASET_DIR/$subject"

  python main.py \
  --name $subject \
  --base ./configs/perfusion_custom.yaml \
  --basedir ../../logs/subjects/perfusion \
  -t True \
  --gpus 7, \
  model.params.personalization_config.params.initializer_words="['$class']" \
  data.params.train.params.data_root="$INSTANCE_DIR" \

done