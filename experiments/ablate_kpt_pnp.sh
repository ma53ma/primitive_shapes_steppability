cd src 

DIST_TH_LIST=(0.01 0.02 0.03)
ANGLE_TH_LIST=(20 30 45)

for KPT_TYPE in box hedron tail; do
	for PNP_ALG in cvIPPE cvP3P cvEPnP; do
		echo ""
		echo ""
		echo "====================================== KPT_TYPE="${KPT_TYPE}"; PNP_ALG="${PNP_ALG}"========================================="
		if [[ $KPT_TYPE == hedron && $PNP_ALG == cvIPPE  ]]; then
			echo " This combination is N/A. Skip."
			continue	
		fi
		# the threshold
		for (( i=0;i<${#DIST_TH_LIST[@]};i++)); do
			DIST_TH=${DIST_TH_LIST[i]}	
			ANGLE_TH=${ANGLE_TH_LIST[i]}	
			echo "----------- The threshold: Distance_Th = "${DIST_TH}"; Angle_Th = "${ANGLE_TH}" ---------------"
			python test.py grasp_pose --exp_id grasp_test_1k_${KPT_TYPE}_${PNP_ALG} --dataset ps_grasp --keep_res --load_model ../exp/grasp_pose/grasp_train_single_1k_${KPT_TYPE}/model_last.pth --not_prefetch_test --trainval --kpt_type ${KPT_TYPE} --pnp_type ${PNP_ALG} --center_thresh 0.1 --vis_thresh 0.1 --kpts_hm_thresh 0.2 --open_width_canonical 0.1 --no_nms --dist_th ${DIST_TH} --angle_th ${ANGLE_TH} --rot_sample_num 30 --trl_sample_num 10 --ori_clf_mode --ori_num 9
		done
	done
done
