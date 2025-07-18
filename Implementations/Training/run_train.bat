python train.py ^
--baseroot "dataset" ^
--save_path "./models" ^
--sample_path "./samples" ^
--gpu_ids "0,1" ^
--gan_type "WGAN" ^
--cudnn_benchmark True ^
--checkpoint_interval 1 ^
--multi_gpu True ^
--load_name "" ^
--epochs 41 ^
--batch_size 2 ^
--lr_g 1e-4 ^
--lr_d 1e-4 ^
--lambda_l1 10 ^
--lambda_perceptual 10 ^
--lambda_gan 1 ^
--lr_decrease_epoch 10 ^
--lr_decrease_factor 0.5 ^
--num_workers 8 ^
--in_channels 4 ^
--out_channels 3 ^
--latent_channels 48 ^
--pad_type "zero" ^
--activation "elu" ^
--norm "none" ^
--init_type "kaiming" ^
--init_gain 0.02 ^
--imgsize 256 ^
--mask_type "free_form" ^
--margin 10 ^
--mask_num 20 ^
--bbox_shape 30 ^
--max_angle 4 ^
--max_len 40 ^
--max_width 10 ^
--pretrained_model_path "pretrained_model/deepfillv2_WGAN_G_epoch40_batchsize4.pth"