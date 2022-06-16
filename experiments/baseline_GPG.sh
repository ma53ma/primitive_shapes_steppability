cd src

#DIST_TH_LIST=(0.01 0.02 0.03)
#ANGLE_TH_LIST=(20 30 45)

DIST_TH_LIST=(0.01)
ANGLE_TH_LIST=(20)

for (( i=0;i<${#DIST_TH_LIST[@]};i++ ));do
	DIST_TH=${DIST_TH_LIST[i]}
	ANGLE_TH=${ANGLE_TH_LIST[i]}
	echo ""
	echo "----------- The threshold: Distance_Th = "${DIST_TH}"; Angle_Th = "${ANGLE_TH}" ---------------"
	python test_baselines.py \
		grasp_baseline_GPG \
		--exp_id grasp_test_GPG \
		--baseline_method GPG \
		--using_mp \
		--dataset ps_grasp \
		--rot_sample_num 30 \
		--trl_sample_num 10 \
		--dist_th ${DIST_TH} \
		--angle_th ${ANGLE_TH} \
		#--load_exist_baseResults
done

