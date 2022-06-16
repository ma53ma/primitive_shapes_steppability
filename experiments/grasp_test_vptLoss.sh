cd src

# get the arguments
DATASET_NUM=${1}
CENTER_TH=${2}
KPT_TH=${3}
VPT_COEFF=${4}

# optional argument: the sample number
ROT_SAM_NUM=-1
TRL_SAM_NUM=-1
if [ ! -z "${5}" ]
then
	ROT_SAM_NUM=${5}
fi
if [ ! -z "${6}" ]
then
	TRL_SAM_NUM=${6}
fi

# parse the suffixs
CENTER_SUFFIX=dot${CENTER_TH##*.}
if [ `echo "${KPT_TH} < 1" | bc` -eq 1 ]
then
	KPT_SUFFIX=dot${KPT_TH##*.}
else
	KPT_SUFFIX=no
fi



python test.py grasp_pose \
	--exp_id grasp_test_${DATASET_NUM}_vpt${VPT_COEFF}_RL_${CENTER_SUFFIX}_${KPT_SUFFIX} \
	--dataset ps_grasp \
	--keep_res \
	--load_model ../exp/grasp_pose/grasp_train_single_${DATASET_NUM}_vptNeg${VPT_COEFF}_RL/model_last.pth \
	--not_prefetch_test \
	--trainval \
	--kpt_type box \
	--pnp_type cvIPPE \
	--center_thresh ${CENTER_TH} \
	--vis_thresh ${CENTER_TH} \
	--kpts_hm_thresh ${KPT_TH} \
	--open_width_canonical 0.1 \
	--no_nms \
	--rot_sample_num ${ROT_SAM_NUM} \
	--trl_sample_num ${TRL_SAM_NUM} \
	# --use_center
	# --reproj_error_th 1	# The reprojection error threshold
