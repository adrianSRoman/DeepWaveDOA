'''
Run the pre-trained CDBPN model on multichannel audio.

Takes .wav files and, for each one, computes the 4-channel spatial covariance
(visibility) matrices and the 32-channel matrices predicted by CDBPN. The
feature extraction matches the setup used to train checkpoints/cdbpn_epoch_99.pth:
  * 16 mel-spaced bands, librosa.mel_frequencies(n_mels=16, fmin=50, fmax=4500)
  * 50 Hz bandwidth per band
  * 10 ms STFT frames, summed over 100 ms blocks -> 10 matrices per second
  * Each matrix is normalized so that its largest eigenvalue is 1

Input wav files can be either:
  * 32 channels (Eigenmike em32). The 4-channel input is taken from capsules
    6, 10, 26, 22 (0-indexed [5, 9, 25, 21]) and the true 32-channel matrices
    are also saved so they can be compared to the prediction.
  * 4 channels in the DCASE/STARSS "MIC" format (the same four em32 capsules,
    in that order).

Example:
    python inference_cdbpn.py --input path/to/wavs --output_dir cdbpn_out \
        --checkpoint ../checkpoints/cdbpn_epoch_99.pth
'''

import argparse
import glob
import os

import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal.windows as windows
import skimage.util as skutil
import torch

from cdbpn import Net as CDBPN

# Feature settings used to train checkpoints/cdbpn_epoch_99.pth. Do not change
# them unless you retrain the model.
NBANDS = 16
FMIN, FMAX = 50.0, 4500.0
BW = 50.0
T_STI = 10.0e-3
T_STATIONARITY = 10 * T_STI
EM32_TO_MIC = [5, 9, 25, 21]  # em32 capsules 6, 10, 26, 22 (DCASE MIC format)


def band_center_frequencies():
    return librosa.mel_frequencies(n_mels=NBANDS, fmin=FMIN, fmax=FMAX)


def extract_visibilities(data, rate, T, fc, bw, alpha):
    '''
    Covariance (visibility) matrix per T-second frame for one band.
    Returns a (N_frames, N_channel, N_channel) complex array.
    '''
    N_stft_sample = int(rate * T)
    N_sample = (data.shape[0] // N_stft_sample) * N_stft_sample
    N_channel = data.shape[1]
    stf_data = (skutil.view_as_blocks(data[:N_sample], (N_stft_sample, N_channel))
                .squeeze(axis=1))  # (N_stf, N_stft_sample, N_channel)

    window = windows.tukey(M=N_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stft_data = np.fft.fft(stf_data * window, axis=1)
    N_stf = stft_data.shape[0]

    # Sum the FFT bins inside the band
    idx_start = int((fc - 0.5 * bw) * N_stft_sample / rate)
    idx_end = int((fc + 0.5 * bw) * N_stft_sample / rate)
    collapsed_spectrum = np.sum(stft_data[:, idx_start:idx_end + 1, :], axis=1)

    return (collapsed_spectrum.reshape(N_stf, -1, 1).conj() *
            collapsed_spectrum.reshape(N_stf, 1, -1))


def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    '''
    Sums T_sti-second matrices over T_stationarity-second blocks.
    Returns a (N_blocks, N_channel, N_channel) complex array.
    '''
    S_sti = extract_visibilities(data, rate, T_sti, fc, bw, alpha=1.0)
    N_channel = data.shape[1]
    N_sti_per_stationary_block = int(round(T_stationarity / T_sti))
    return (skutil.view_as_windows(S_sti,
                                   (N_sti_per_stationary_block, N_channel, N_channel),
                                   (N_sti_per_stationary_block, N_channel, N_channel))
            .squeeze(axis=(1, 2))
            .sum(axis=1))


def normalize_max_eigenvalue(S):
    '''
    Rescale each matrix so its largest eigenvalue is 1 and drop negative
    eigenvalues. All-zero (silent) frames stay zero. S: (..., N, N).
    '''
    S_D, S_V = np.linalg.eigh(S)
    S_max = S_D.max(axis=-1, keepdims=True)
    S_D = np.where(S_max > 0, np.clip(S_D / np.where(S_max > 0, S_max, 1), 0, None), 0)
    return (S_V * S_D[..., None, :]) @ S_V.conj().swapaxes(-1, -2)


def get_visibility_matrix(audio, fs):
    '''
    audio: (N_sample, N_channel) array.
    Returns a (N_frames, NBANDS, N_channel, N_channel) complex128 array.
    '''
    vis = [normalize_max_eigenvalue(form_visibility(audio, fs, fc, BW, T_STI, T_STATIONARITY))
           for fc in band_center_frequencies()]
    return np.stack(vis, axis=1)


def load_audio(path, input_format):
    fs, audio = wavfile.read(path)
    if audio.ndim == 1:
        raise ValueError(f'{path}: expected multichannel audio, got mono')
    audio = audio.astype(np.float64)  # absolute scale does not matter, see normalize_max_eigenvalue

    nch = audio.shape[1]
    if input_format == 'auto':
        input_format = {32: 'em32', 4: 'mic'}.get(nch)
    if input_format == 'em32' and nch == 32:
        return audio[:, EM32_TO_MIC], audio, fs
    if input_format == 'mic' and nch == 4:
        return audio, None, fs
    raise ValueError(f'{path}: has {nch} channels, expected 32 (em32) or 4 (MIC format)')


def load_model(checkpoint, device):
    model = CDBPN(num_channels=NBANDS, base_filter=32, feat=128, num_stages=10, scale_factor=8)
    state_dict = torch.load(checkpoint, map_location='cpu')
    # The checkpoint was saved from nn.DataParallel, so keys start with 'module.'
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.no_grad()
def upsample(model, vis4, device, batch_size):
    '''vis4: (N_frames, NBANDS, 4, 4) -> (N_frames, NBANDS, 32, 32), both complex.'''
    out = []
    for i in range(0, vis4.shape[0], batch_size):
        x = torch.from_numpy(vis4[i:i + batch_size]).to(device)
        out.append(model(x.real.double(), x.imag.double()).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    parser = argparse.ArgumentParser(description='CDBPN inference: 4ch -> 32ch covariance matrices')
    parser.add_argument('--input', required=True, help='A .wav file or a folder of .wav files')
    parser.add_argument('--output_dir', default='cdbpn_out', help='Where to write one .npz per input file')
    parser.add_argument('--checkpoint', default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'cdbpn_epoch_99.pth'))
    parser.add_argument('--input_format', choices=['auto', 'em32', 'mic'], default='auto',
                        help='em32: 32ch Eigenmike, mic: 4ch DCASE MIC format, auto: decide from channel count')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of frames per forward pass')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    opt = parser.parse_args()

    if os.path.isdir(opt.input):
        files = sorted(glob.glob(os.path.join(opt.input, '*.wav')))
    else:
        files = [opt.input]
    if not files:
        raise SystemExit(f'No .wav files found in {opt.input}')
    os.makedirs(opt.output_dir, exist_ok=True)

    model = load_model(opt.checkpoint, opt.device)
    freqs = band_center_frequencies()
    print(f'Band center frequencies [Hz]: {np.round(freqs, 1)}')

    for path in files:
        mic, em32, fs = load_audio(path, opt.input_format)
        vis4 = get_visibility_matrix(mic, fs)
        vis32_pred = upsample(model, vis4, opt.device, opt.batch_size)

        result = dict(vis4=vis4, vis32_pred=vis32_pred, freqs=freqs, fs=fs)
        msg = f'{os.path.basename(path)}: {vis4.shape[0]} frames, vis4 {vis4.shape} -> vis32_pred {vis32_pred.shape}'
        if em32 is not None:
            result['vis32_true'] = get_visibility_matrix(em32, fs)
            mae = np.abs(vis32_pred - result['vis32_true']).mean()
            msg += f', mean abs error vs. true 32ch: {mae:.4f}'

        out_path = os.path.join(opt.output_dir, os.path.splitext(os.path.basename(path))[0] + '.npz')
        np.savez_compressed(out_path, **result)
        print(f'{msg}\n  saved to {out_path}')


if __name__ == '__main__':
    main()
