# now ready tasks groceries pour_tea stack_basket build_blocks organize_pants pour_bowl wipe_stains seal_bag stir_coffee scoop_bean

# save_data_root = /mnt/hpfs/baaiei/qianpusun/data/lerobot_data/agilex

# rm -rf /mnt/hpfs/baaiei/qianpusun/data/lerobot_data/agilex/*
export http_proxy=http://192.168.0.3:1080 
export https_proxy=http://192.168.0.3:1080
# /mnt/hpfs/baaiei/chenliming/data/lerobot_data
# /mnt/hpfs/baaiei/robot_data/pika/pour_tea
OBJECT=/mnt/hpfs/baaiei/robot_data/pika/pour_tea
NEW_OBJECT=$(basename "$OBJECT")
MACHINE=pika
cd /mnt/hpfs/baaiei/lvhuaihai/openpi/

# uv run /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/agilex_process/convert_mixed_multidir_to_lerobot.py \
#     --raw_dirs $OBJECT \
#     --repo_id=HuaihaiLyu/${MACHINE}_${NEW_OBJECT} \
#     --mode="video" 

uv run /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/pika_process/convert_mixed_multidir_to_lerobot.py \
    --raw_dirs $OBJECT \
    --repo_id=HuaihaiLyu/${MACHINE}_${NEW_OBJECT} \
    --mode="video" 

# mv /mnt/hpfs/baaiei/chenliming/data/lerobot_data/HuaihaiLyu/${MACHINE}_${NEW_OBJECT} /mnt/hpfs/baaiei/qianpusun/data/lerobot_data/HuaihaiLyu/

# bash /mnt/hpfs/baaiei/lvhuaihai/openpi/examples/aloha_real/agilex_process/process.sh
