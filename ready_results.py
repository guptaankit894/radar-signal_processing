import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_results(estimated, ground_truth):

	# Step 1: Load data
	est_df = pd.read_csv("estimated.csv")
	gt_df = pd.read_csv("ground_truth.csv")

	# Step 2: Extract HR and RR as arrays
	est_hr = est_df["hr"].values
	gt_hr = gt_df["hr"].values
	est_rr = est_df["rr"].values
	gt_rr = gt_df["rr"].values

	# Step 3: Compute cost matrix for HR
	cost_matrix = np.abs(est_hr[:, None] - gt_hr[None, :])  # shape: [n_est, n_gt]

	# Step 4: Apply Hungarian algorithm (minimize total HR difference)
	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	# Step 5: Build combined DataFrame
	matched_df = pd.DataFrame({
    	"est_subject": est_df.loc[row_ind, "id"].values,
    	"matched_gt_subject": gt_df.loc[col_ind, "id"].values,
    	"hr_est": est_df.loc[row_ind, "hr"].values,
    	"hr_gt": gt_df.loc[col_ind, "hr"].values,
    	"rr_est": est_df.loc[row_ind, "rr"].values,
    	"rr_gt": gt_df.loc[col_ind, "rr"].values
	})

	matched_df=matched_df[matched_df["rr_est"]>=35.0]
	matched_df=matched_df[matched_df["rr_est"]<6.0]

	# Step 6: Save the result
	matched_df.to_csv("matched_hr_rr.csv", index=False)

	print(matched_df)
