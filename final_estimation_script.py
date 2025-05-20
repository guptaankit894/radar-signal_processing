import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, hilbert
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks, hilbert
import scipy
from scipy.optimize import linear_sum_assignment
import os


# Function Definitions
# -- 1. Decode raw BIN → ADC cube (x  chirps, x adc samples, we have to variate according to the file) --
def decode_iwr6843_data(filename,
                        num_rx=4,
                        num_adc_samples=256,
                        num_chirps=64,
                        header_bytes=0):
    with open(filename, 'rb') as f:
        f.seek(header_bytes)
        raw = np.fromfile(f, dtype=np.int16)
    raw = raw.reshape(-1, 2)
    complex_data = raw[:,0] + 1j*raw[:,1]
    samp_pf = num_rx * num_chirps * num_adc_samples
    num_frames = len(complex_data) // samp_pf

    adc = np.empty((num_frames, num_rx, num_chirps, num_adc_samples),
                   dtype=complex)
    for fr in range(num_frames):
        for rx in range(num_rx):
            st = fr*samp_pf + rx*(num_chirps*num_adc_samples)
            en = st + (num_chirps*num_adc_samples)
            adc[fr,rx] = complex_data[st:en].reshape(num_chirps, num_adc_samples)
    return adc
def compute_range_profiles(adc):
    return np.fft.fft(adc, axis=-1)

# -- 3. Range–Angle map -- (variate n_angle_bins according to the chirps)
def compute_range_angle_map(rp, n_angle_bins=64):
    avg_rx = rp.mean(axis=(0,2))
    ang = np.fft.fftshift(
        np.fft.fft(avg_rx, n=n_angle_bins, axis=0),
        axes=0
            )
    return np.abs(ang).T

# -- 4. Physical‐distance‐based target pick -- (variate n_angle_bins according to the adc samples)
def detect_targets_physical(ra_map, num_targets=3,
                            min_dist_m=0.5, B=3.9e9,
                            n_angle_bins=256, R_mean=4.0):
    c = 3e8
    rng_res = c/(2*B)
    min_rb = int(np.ceil(min_dist_m/rng_res))
    ang_res = 180.0/n_angle_bins
    ang_sep = np.degrees(np.arcsin(min_dist_m/R_mean))
    min_ab = int(np.ceil(ang_sep/ang_res))

    R, A = ra_map.shape
    idxs = np.argsort(ra_map.flatten())[::-1]
    targets = []
    for idx in idxs:
        r, a = divmod(idx, A)
        if r < min_rb or r > R - min_rb:
            continue
        if any(abs(r-r0)<min_rb and abs(a-a0)<min_ab for r0,a0 in targets):
            continue
        targets.append((r,a))
        if len(targets) == num_targets:
            break
    return targets

# -- 5. MVDR beamforming ±3‐bin slab --(variate n_angle_bins according to the adc samples)
def separate_with_mvdr(adc, targets,
                       n_angle_bins,
                       fc=61e9,
                       rng_win=3,
                       eps=1e-6):
    frames, num_rx, chirps, samples = adc.shape
    rp = np.fft.fft(adc, axis=-1)
    rng_bins = rp.shape[-1]
    angle_axis = np.linspace(-90, 90, n_angle_bins)

    def steer(theta):
        lam = 3e8/fc
        d = lam/2
        k = 2*np.pi/lam
        idx = np.arange(num_rx)
        a = np.exp(-1j*k*d*idx*np.sin(np.deg2rad(theta)))
        return a[:,None]

    beams = np.zeros((frames, len(targets)), dtype=complex)
    for i, (rbin, abin) in enumerate(targets):
        lo = max(0, rbin-rng_win)
        hi = min(rng_bins, rbin+rng_win+1)
        gated = rp[:,:,:,lo:hi]
        X = gated.transpose(1,0,2,3).reshape(num_rx, -1)
        Rcov = X @ X.conj().T / X.shape[1] + eps*np.eye(num_rx)
        a_vec = steer(angle_axis[abin])
        w = np.linalg.inv(Rcov) @ a_vec
        w /= (a_vec.conj().T @ np.linalg.inv(Rcov) @ a_vec)
        for fr in range(frames):
            Y = gated[fr].reshape(num_rx, -1)
            bf = (w.conj().T @ Y).reshape(-1)
            beams[fr, i] = bf.sum()
    return beams
# -- 6. Filters & HR/RR extraction --
def bandpass_filter(x, low, high, fs, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

def estimate_rate_welch(x, fs, fmin, fmax):
    f, P = welch(x, fs=fs, nperseg=min(len(x), int(5*fs)))
    mask = (f>=fmin) & (f<=fmax)
    return (f[mask][np.argmax(P[mask])] * 60.0) if np.any(mask) else np.nan


results=pd.DataFrame(columns=['id','hr','rr'])
count=0
for idx in glob.glob("*.bin"):
	temp=idx.split("_")
	person_name=temp[0]
	num_person = re.search(r'MAN(\d+)', idx, flags=re.IGNORECASE)
	ADC = re.search(r'ADC(\d+)', idx, flags=re.IGNORECASE)
	chirp = re.search(r'chirp(\d+)', idx, flags=re.IGNORECASE)
	FR=re.search(r'FR(\d+)', idx, flags=re.IGNORECASE)
	print(idx)
	print(r"num_person:{}, ADC:{}, chirp:{}, FP:{}".format(num_person.group(1),ADC.group(1),chirp.group(1), FR.group(1)))

	# Main processing & visualization
	
	adc = decode_iwr6843_data(idx)
	rp  = compute_range_profiles(adc)
	ra_map = compute_range_angle_map(rp, n_angle_bins=int(chirp.group(1)))
	targets = detect_targets_physical(ra_map, int(num_person.group(1)), min_dist_m=0.5, B=3.9e9, n_angle_bins=int(ADC.group(1)), R_mean=4.0) #(variate n_angle_bins according to the adc samples)
	beams = separate_with_mvdr(adc, targets, n_angle_bins=int(ADC.group(1)), rng_win=3)

	FS = int(FR.group(1))  # frame rate

	for i in range(beams.shape[1]):
		z= beams[:, i]
		phase = np.unwrap(np.angle(z))
		# Heart rate: phase derivative → Welch
		dphi  = np.gradient(phase) * FS / (2*np.pi)
		heart = bandpass_filter(dphi, 1.0, 3.0, FS)
		hr    = estimate_rate_welch(heart, FS, 1.0, 3.0)

		# Respiration rate: peak count on instantaneous frequency
		instf = dphi  # instantaneous frequency already
		instf_bp = bandpass_filter(instf, 0.1, 0.6, FS)
		min_dist = int(FS * 60.0 / 50)
		prom     = 0.3 * np.std(instf_bp)
		peaks, _ = find_peaks(instf_bp, distance=min_dist, prominence=prom)
		rr = len(peaks) / (len(instf_bp)/(FS*60.0))

		# Plot both signals
		t = np.arange(len(z)) / FS
		plt.figure(figsize=(10, 5))
		plt.subplot(2,1,1)
		plt.plot(t, instf_bp, label='Resp InstF (0.1–0.6 Hz)')
		plt.plot(peaks/FS, instf_bp[peaks], 'ro')
		plt.title(f'Patient {i+1} Respiration Signal (RR={rr:.1f} bpm)')
		plt.xlabel('Time (s)'); plt.ylabel('Freq (Hz)')
		plt.legend()

		plt.subplot(2,1,2)
		plt.plot(t, heart, label='Heart InstF (1–3 Hz)')
		plt.title(f'Patient {i+1} Heart Signal (HR={hr:.1f} bpm)')
		plt.xlabel('Time (s)'); plt.ylabel('Freq (Hz)')
		plt.legend()
		plt.tight_layout()
		plt.savefig('./imgs/'+person_name+'_'+str(i)+'.png')
		#plt.show()
		#print(f'Patient {i+1}: RR = {rr:.1f} bpm, HR = {hr:.1f} bpm')
		results.loc[count,'id']=person_name
		#results.loc[count,'subject']=person_name
		results.loc[count,'hr']=hr
		results.loc[count,'rr']=rr
		count=count+1


#results = results[results['rr'] <= 6.0 and results['rr']>=35.0] 
results.to_csv("estimated.csv", index=False)

#################################Get final data######################
def get_results(estimated, ground_truth):

	# Step 1: Load data
	est_df = pd.read_csv(estimated)
	gt_df = pd.read_csv(ground_truth)

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

	matched_df=matched_df[matched_df['rr_est']<=35.0]
	matched_df=matched_df[matched_df['rr_est']>6.0]

	# Step 6: Save the result
	matched_df.to_csv("matched_hr_rr.csv", index=False)

	print(matched_df)


get_results("estimated.csv", "ground_truth.csv")	
	
	


