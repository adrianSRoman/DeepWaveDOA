import librosa
import numpy as np
import os
import math
import pickle
from pysofaconventions import *


from scipy.io import wavfile


import os
from sklearn import preprocessing
import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
plot.switch_backend('agg')
import shutil
import math
import wave
import contextlib

import scipy.constants as constants
import scipy.io.wavfile as wavfile
import scipy.signal.windows as windows
import scipy.linalg as linalg
import skimage.util as skutil
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import scipy.spatial as spatial

import torch
from DBPN import Net as DBPNCX

from matplotlib.ticker import FuncFormatter

FS = 48000 # original impulse reponse sampling rate
NEW_FS = 24000 # new sampling rate (same as DCASE Synth)

def extract_visibilities(_data, _rate, T, fc, bw, alpha):
    """
    Transform time-series to visibility matrices.

    Parameters
    ----------
    T : float
        Integration time [s].
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    alpha : float
        Shape parameter of the Tukey window, representing the fraction of
        the window inside the cosine tapered region. If zero, the Tukey
        window is equivalent to a rectangular window. If one, the Tukey
        window is equivalent to a Hann window.

    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices (complex-valued).
    """
    N_stft_sample = int(_rate * T)
    if N_stft_sample == 0:
        raise ValueError('Not enough samples per time frame.')
    # print(f'Samples per STFT: {N_stft_sample}')

    N_sample = (_data.shape[0] // N_stft_sample) * N_stft_sample
    N_channel = _data.shape[1]
    stf_data = (skutil.view_as_blocks(_data[:N_sample], (N_stft_sample, N_channel))
                .squeeze(axis=1))  # (N_stf, N_stft_sample, N_channel)

    window = windows.tukey(M=N_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stf_win_data = stf_data * window  # (N_stf, N_stft_sample, N_channel)
    N_stf = stf_win_data.shape[0]

    stft_data = np.fft.fft(stf_win_data, axis=1)  # (N_stf, N_stft_sample, N_channel)
    # Find frequency channels to average together.
    idx_start = int((fc - 0.5 * bw) * N_stft_sample / _rate)
    idx_end = int((fc + 0.5 * bw) * N_stft_sample / _rate)
    collapsed_spectrum = np.sum(stft_data[:, idx_start:idx_end + 1, :], axis=1)

    # Don't understand yet why conj() on first term?
    # collapsed_spectrum = collapsed_spectrum[0,:]
    S = (collapsed_spectrum.reshape(N_stf, -1, 1).conj() *
        collapsed_spectrum.reshape(N_stf, 1, -1))
    return S

def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    '''
    Parameter
    ---------
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) antenna samples. (float)
    rate : int
        Sample rate [Hz]
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    T_sti : float
        Integration time [s]. (time-series)
    T_stationarity : float
        Integration time [s]. (visibility)
        
    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices.
        
        # N_slot == number of audio frames in track

    Note
    ----
    Visibilities computed directly in the frequency domain.
    For some reason visibilities are computed correctly using
    `x.reshape(-1, 1).conj() @ x.reshape(1, -1)` and not the converse.
    Don't know why at the moment.
    '''
    S_sti = (extract_visibilities(data, rate, T_sti, fc, bw, alpha=1.0))

    N_sample, N_channel = data.shape
    N_sti_per_stationary_block = int(T_stationarity / T_sti)
    S = (skutil.view_as_windows(S_sti,
                                (N_sti_per_stationary_block, N_channel, N_channel),
                                (N_sti_per_stationary_block, N_channel, N_channel))
        .squeeze(axis=(1, 2))
        .sum(axis=1))
    return S

def _get_visibility_matrix_for_signal(audio_data, mic_type="em32"):
    # audio_in, fs = _load_audio_deepwave(audio_filename)
    fs = FS
    audio_in = audio_data
    print("Audio in shape", audio_in.shape)
    if "mic" in mic_type:
        audio_in = audio_in[:, [5,9,25,21]]
    else: # em32
        pass # eigenmike default
    
    freq, bw = (skutil  # Center frequencies to form images
        .view_as_windows(np.linspace(50, 9000, 16), (2,), 1)
        .mean(axis=-1)), 50.0  # [Hz]

    visibilities = []
    for i in range(15):
        T_sti = 10.0e-3
        T_stationarity = 10 * T_sti  # Choose to have frame_rate = 10
        S = form_visibility(audio_in, fs, freq[i], bw, T_sti, T_stationarity)
        N_sample = S.shape[0]
        visibilities_per_frame = []
        for s_idx in range(N_sample):
            S_D, S_V = linalg.eigh(S[s_idx])
            if S_D.max() <= 0:
                S_D[:] = 0
            else:
                S_D = np.clip(S_D / S_D.max(), 0, None)
            S_norm = (S_V * S_D) @ S_V.conj().T
            visibilities_per_frame.append(S_norm) 

        visibilities.append(visibilities_per_frame)

    return np.array(visibilities)

def unit_vector(azimuth, elevation):
    """
    Compute unit vector given the azimuth and elevetion of source in 3D space
    Args:
        azimuth (float)
        elevation (float)
    Returns:
        A list representing the coordinate points xyz in 3D space
    """
    x = math.cos(elevation) * math.cos(azimuth)
    y = math.cos(elevation) * math.sin(azimuth)
    z = math.sin(elevation)
    return [x, y, z]

def compute_azimuth_elevation(receiver_pos, source_pos):
    # Calculate the vector from the receiver to the source
    vector = [source_pos[0] - receiver_pos[0], source_pos[1] - receiver_pos[1], source_pos[2] - receiver_pos[2]]
    # Calculate the azimuth angle
    azimuth = math.atan2(vector[0], vector[1])
    # if azimuth < 0:
    #     azimuth += math.pi
    # Calculate the elevation angle
    distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    elevation = math.asin(vector[2] / distance)
    return azimuth, elevation, distance

DIRECT_IR_IDX = 10
RIR_DB = '/scratch/data/RIR_datasets/6dof_SRIRs_eigenmike_raw'
sofa_file = "6DoF_SRIRs_eigenmike_raw_100percent_absorbers_enabled.sofa"
sofa = SOFAFile(os.path.join(RIR_DB, sofa_file),'r')
sourcePositions = sofa.getVariableValue('SourcePosition') # get sound source position
listenerPosition = sofa.getVariableValue('ListenerPosition') # get mic position
# get RIR data
rirdata = sofa.getDataIR()
num_meas, num_ch = rirdata.shape[0], rirdata.shape[1]
meas_per_mic = 3 # equal the number of meas per trajectory
num_meas = 15 # set num_meas to 15 to keep south mics only
angles_mic_src = [math.degrees(compute_azimuth_elevation(lis, src)[0]) \
                    for lis, src in zip(listenerPosition[:num_meas], sourcePositions[:num_meas])]
meas_sorted_ord = np.argsort(angles_mic_src)[::-1]
print(rirdata.shape)
irdata = rirdata[DIRECT_IR_IDX, :, :]
print(irdata.shape)
print(irdata.shape, irdata.shape[1]/FS)

import matplotlib.pyplot as plt

# print("data shape", ir_dat.shape)
ir_dat = irdata[:,:FS//10]
wavfile.write("figures/ir_sig.wav", FS, ir_dat.T)

plt.plot(ir_dat[0])
plt.savefig("ir_plot.png")

vgmat_em32 = _get_visibility_matrix_for_signal(ir_dat.T, mic_type="em32")
print("shape of vgmat (em32)", vgmat_em32.shape)

vgmat_mic = _get_visibility_matrix_for_signal(ir_dat.T, mic_type="mic")
print("shape of vgmat (mic)", vgmat_mic.shape)


vgmic = torch.tensor(vgmat_mic).to('cuda').cfloat()
vgmic = torch.transpose(vgmic, 1, 0)
# vgmic = vgmic.unsqueeze(0)
print("Shape of vgmic", vgmic.shape)


cdbpn = DBPNCX(num_channels=15, base_filter=32,  feat = 128, num_stages=10, scale_factor=8).to('cuda')
pretrained_dict = torch.load('/home/asroman/repos/DBPN-Pytorch/weights/cdbpn_epoch_99.pth', map_location=torch.device('cuda'))
new_pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
cdbpn.load_state_dict(new_pretrained_dict)

out_vg_32 = cdbpn(vgmic.real.double(), vgmic.imag.double())
print("shape of out_vg_32", out_vg_32.shape)

label = np.mean(np.abs(vgmat_em32), axis=0)
pred = np.mean(np.abs(out_vg_32[0].cpu().detach().numpy()), axis=0)
plt.figure()
pixel_plot1 = plt.imshow(label[0], interpolation='nearest')
plt.savefig("figures/em32_target_mag_mean.png")
plt.figure()
pixel_plot2 = plt.imshow(pred, interpolation='nearest')
plt.savefig("figures/em32_pred_mag_mean.png")

label = np.mean(np.angle(vgmat_em32), axis=0)
pred = np.mean(np.angle(out_vg_32[0].cpu().detach().numpy()), axis=0)
plt.figure()
pixel_plot1 = plt.imshow(label[0], interpolation='nearest')
plt.savefig("figures/em32_target_phase_mean.png")
plt.figure()
pixel_plot2 = plt.imshow(pred, interpolation='nearest')
plt.savefig("figures/em32_pred_phase_mean.png")

# Create a custom tick formatter to display radians
def radians_formatter(x, pos):
    degrees = np.degrees(x)  # Convert radians to degrees
    return f'{degrees:.0f}°'

mag_absdiff = []
phs_absdiff = []
for i in range(15):
    vg_pred_mag = np.abs(out_vg_32[0,i,:,:].cpu().detach().numpy())
    vg_labl_mag = np.abs(vgmat_em32[i,:,:])
    print("shape of", np.abs(vg_pred_mag-vg_labl_mag).shape)
    mag_absdiff.append(np.abs(vg_pred_mag-vg_labl_mag))

    vg_pred_phs = np.angle(out_vg_32[0,i,:,:].cpu().detach().numpy())
    vg_labl_phs = np.angle(vgmat_em32[i,:,:])
    diff_phase = vg_labl_phs-vg_pred_phs
    norm_diff_phase = np.arctan(np.sin(diff_phase), np.cos(diff_phase))
    # print("shape of", np.abs(phi_vg_labl_phs-phi_vg_pred_phs).shape)
    print(norm_diff_phase.min(), norm_diff_phase.max())
    # print(np.unwrap(vg_labl_phs).min(), np.unwrap(vg_labl_phs).max())
    phs_absdiff.append(np.abs(norm_diff_phase))
        #np.abs(phi_vg_labl_phs-phi_vg_pred_phs))

freq_bands = np.linspace(50, 9000, 16)
# Calculate the center value for each range
frequency_bands = [(freq_bands[i] + freq_bands[i+1]) / 2 for i in range(len(freq_bands) - 1)]
# Create the plot
plt.figure()
# plt.plot(frequency_bands, phs_mse, label='Phase MSE')
print([str(fq+1) for fq in range(15)])
print(len([mag.mean() for mag in mag_absdiff]))
plt.bar([str(fq+1) for fq in range(15)], [mag.mean() for mag in mag_absdiff], label='Avg. mag abs diff')
plt.xlabel('Frequency Band (1500 - 4500 Hz)')
plt.ylabel('Avg. mag abs diff')
plt.title('Absolute Magnitude Difference (per freq band)')
# plt.xticks(frequency_bands)
# plt.legend()
# plt.grid(True)
plt.savefig("figures/mag_plot_bands.png")

# Create the plot
plt.figure()
tick_positions = [ 0, np.pi/4, np.pi/2]
# plt.plot(frequency_bands, phs_mse, label='Phase MSE')
plt.bar([str(fq+1) for fq in range(15)], [phs.mean()*(180 / np.pi) for phs in phs_absdiff], label='Avg. phase diff')
plt.xlabel('Frequency Band (1500 - 4500 Hz)')
plt.ylabel('Avg. phase diff (degrees)')
plt.title('Phase Difference (per freq band)')
# plt.xticks(frequency_bands)
# plt.yticks(tick_positions)
# plt.legend()
# plt.grid(True)
plt.savefig("figures/phase_plot_bands.png")

for i in range(9):
    plt.figure()
    pixel_plot1 = plt.imshow(mag_absdiff[i][0], interpolation='nearest', cmap="YlOrRd")
    plt.colorbar()  # This adds the colorbar to the plot
    plt.title('Magnitude Absolute Difference')
    plt.savefig(f"figures/mag_diff_freq{i}.png")

tick_positions = [ 0, np.pi/4, np.pi/2]
tick_labels = ['0', 'π/4', 'π/2']
for i in range(9):
    plt.figure()
    pixel_plot1 = plt.imshow(phs_absdiff[i][0], interpolation='nearest', cmap="YlOrRd", vmin=0, vmax=np.pi/2)
    # plt.colorbar(format=FuncFormatter(radians_formatter))  # This adds the colorbar to the plot
    # Set the tick formatter for the color bar
    plt.colorbar(format=FuncFormatter(radians_formatter), ticks=tick_positions)
    plt.title(f'Phase Absolute Difference (freq_band={i})')
    plt.savefig(f"figures/phase_diff_freq{i}.png")


fig, axes = plt.subplots(3, 3, figsize=(32, 32))

# # Flatten the 2D array of axes for easier iteration
axes = axes.flatten()

# Loop through data and axes to create scatterplots
for i, ax in enumerate(axes):
    print(i, ax)
    # ax.scatter(data[i], data[i], marker='o', color='b')
    ax.imshow(mag_absdiff[i][0], interpolation='nearest', cmap="YlOrRd")
plt.savefig("figures/mat_mag.png")
# Adjust layout to prevent overlapping titles
# plt.tight_layout()
# Show the plots
# plt.savefig("figures/mse_mag_phase.png")

# freq_bands = np.linspace(1500, 4500, 10)
# # Calculate the center value for each range
# frequency_bands = [(freq_bands[i] + freq_bands[i+1]) / 2 for i in range(len(freq_bands) - 1)]
# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(frequency_bands, phs_mse, label='Phase MSE')
# plt.plot(frequency_bands, mag_mse, label='Magnitude MSE')
# plt.xlabel('Frequency Band Center')
# plt.ylabel('MSE Error')
# plt.title('MSE Errors for phase and magnitude per freq band')
# plt.xticks(frequency_bands)
# plt.legend()
# plt.grid(True)
# plt.savefig("figures/mse_mag_phase.png")

# vgmic_input = audioMIC_vg[vgIdx]
# input = np.mean(np.abs(vgmic_input), axis=0)
# plt.figure()
# pixel_plot1 = plt.imshow(input, interpolation='nearest')
# plt.savefig("figures/mic_input.png")
