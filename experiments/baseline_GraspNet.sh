cd src

DIST_TH_LIST=(0.01 0.02 0.03)
ANGLE_TH_LIST=(20 30 45)

#DIST_TH_LIST=(0.10)
#ANGLE_TH_LIST=(60)

for (( i=0;i<${#DIST_TH_LIST[@]};i++ ));do
	DIST_TH=${DIST_TH_LIST[i]}
	ANGLE_TH=${ANGLE_TH_LIST[i]}
	echo ""
	echo "----------- The threshold: Distance_Th = "${DIST_TH}"; Angle_Th = "${ANGLE_TH}" ---------------"
	python test_baselines.py \
		grasp_baseline_GraspNet \
		--exp_id grasp_test_GraspNet \
		--baseline_method GraspNet \
		--dataset ps_grasp \
		--rot_sample_num 30 \
		--trl_sample_num 10 \
		--dist_th ${DIST_TH} \
		--angle_th ${ANGLE_TH} \
		--grasp_sampler_folder ./checkpoints/gan_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2 \
		--num_grasp_samples 100 \
		--load_exist_baseResults
done

