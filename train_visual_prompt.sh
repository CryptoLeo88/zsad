device=1

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visual_prompt
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python train_visual_prompt.py --dataset visa --train_data_path /remote-home/iot_zhouqihang/data/Visa \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 518 --batch_size 8 --print_freq 1 \
        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done


LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visual_prompt_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python train_visual_prompt.py --dataset mvtec --train_data_path /remote-home/iot_zhouqihang/data/mvdataset \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 518 --batch_size 8 --print_freq 1 \
        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done
