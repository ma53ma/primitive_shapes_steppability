cd src

DATASET_NUM=${1}
CENTER_TH=${2}
KPT_TH=${3}
ORI_NUM=${4}
VPT_COEFF=${5}

CENTER_SUFFIX=dot${CENTER_TH##*.}
if [ `echo "${KPT_TH} < 1" | bc` -eq 1 ]
then
	KPT_SUFFIX=dot${KPT_TH##*.}
else
	KPT_SUFFIX=no
fi

if [ `echo "${VPT_COEFF} < 0" | bc` -eq 1 ]
then
	EXP_ID=debug_test_${DATASET_NUM}_RL_oriClf${ORI_NUM}_${CENTER_SUFFIX}_${KPT_SUFFIX}
	LOAD_MODEL=../exp/grasp_pose/grasp_train_single_${DATASET_NUM}_RL_oriClf${ORI_NUM}/model_last.pth
else
	EXP_ID=debug_test_${DATASET_NUM}_vpt${VPT_COEFF}_RL_oriClf${ORI_NUM}_${CENTER_SUFFIX}_${KPT_SUFFIX}
	LOAD_MODEL=../exp/grasp_pose/grasp_train_single_${DATASET_NUM}_vptNeg${VPT_COEFF}_RL_oriClf${ORI_NUM}/model_last.pth
fi

python test.py grasp_pose \
	--exp_id ${EXP_ID} \
	--dataset ps_grasp \
	--keep_res \
	--load_model ${LOAD_MODEL} \
	--not_prefetch_test \
	--trainval \
	--kpt_type box \
	--pnp_type cvIPPE \
	--center_thresh ${CENTER_TH} \
	--vis_thresh ${CENTER_TH} \
	--kpts_hm_thresh ${KPT_TH} \
	--open_width_canonical 0.1 \
	--no_nms \
	--ori_clf_mode \
	--ori_num ${ORI_NUM} \
	--debug 5 \
	--test_num_per_shape 30 \
	#--vis_results \
	# --reproj_error_th 1.0 # The reprojection error threshold 
