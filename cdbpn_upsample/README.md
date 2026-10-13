# CDBPN inference: 4-channel → 32-channel covariance matrices

`inference_cdbpn.py` runs the pre-trained CDBPN checkpoint (`../checkpoints/cdbpn_epoch_99.pth`) on your audio. For each `.wav` file it saves:

1. the **4-channel covariance (visibility) matrices** used as input, and
2. the **32-channel covariance matrices predicted** by CDBPN.

If you give it 32-channel Eigenmike recordings, it also computes the **true 32-channel matrices** so you can compare them with the prediction.

## 1. Setup

Python ≥ 3.8 with:

```bash
pip install torch numpy scipy scikit-image librosa
```

A GPU is optional. The script uses CUDA if it is available, and the CPU otherwise.

## 2. Run

```bash
cd cdbpn_upsample
python inference_cdbpn.py --input /path/to/wavs --output_dir cdbpn_out
```

| Argument | Default | Meaning |
|---|---|---|
| `--input` | (required) | One `.wav` file or a folder of `.wav` files |
| `--output_dir` | `cdbpn_out` | One `.npz` file is written per input file |
| `--checkpoint` | `../checkpoints/cdbpn_epoch_99.pth` | Pre-trained weights |
| `--input_format` | `auto` | `em32` (32 ch), `mic` (4 ch), or `auto` (chosen from the channel count) |
| `--batch_size` | `64` | Frames per forward pass. Lower it if you run out of memory |
| `--device` | `cuda` if available | `cuda` or `cpu` |

## 3. Output

```python
import numpy as np
d = np.load("cdbpn_out/my_file.npz")
d["vis4"]        # (T, 16, 4, 4)   complex: input 4-ch matrices
d["vis32_pred"]  # (T, 16, 32, 32) complex: CDBPN prediction
d["vis32_true"]  # (T, 16, 32, 32) complex: true 32-ch matrices (only for 32-ch input)
d["freqs"]       # (16,) band center frequencies [Hz]
d["fs"]          # sample rate of the input file
```

`T` is the number of frames. There are **10 frames per second** (one matrix per 100 ms).

## 4. Things to check with your own data

**Microphone array (most important).** The model has only learned to map **these four Eigenmike em32 capsules** to the full em32 array:

| Input channel | em32 capsule (1-indexed) | azimuth, elevation |
|---|---|---|
| 1 | 6  | 45°, 35° |
| 2 | 10 | −45°, −35° |
| 3 | 26 | 135°, −35° |
| 4 | 22 | −135°, 35° |

This is the tetrahedral **DCASE / STARSS "MIC" format**, with radius 4.2 cm. Your data must:
- be 4-channel MIC-format audio in exactly this channel order, or
- be 32-channel em32 audio in the standard em32 capsule order. The script then picks channels `[5, 9, 25, 21]` (0-indexed) itself.

Recordings from any other array geometry or channel order will produce meaningless outputs. The model would need to be retrained for them.

**The 32-channel output** is in standard em32 capsule order (1 to 32). To beamform or image with it, use the em32 capsule positions.

**Features are fixed by the checkpoint.** Do not change them unless you retrain:
- 16 mel-spaced bands: `librosa.mel_frequencies(n_mels=16, fmin=50, fmax=4500)`, each 50 Hz wide
- 10 ms FFT frames with a Hann window, summed over 100 ms blocks
- each matrix is scaled so its **largest eigenvalue is 1**, and negative eigenvalues are set to 0. Fully silent frames stay all-zero.

Because of this scaling, the absolute signal level (int16 or float, gain) does not matter.

**Sample rate.** The training data was at 24 kHz. The frequency resolution of the features does not depend on the sample rate (bins are always 100 Hz wide). Any rate of 10 kHz or more works, and you do not need to resample.

**Matrix convention.** Entry `S[i, j] = conj(X_i) · X_j`, where `X` is the per-channel spectrum in the band. This is the **complex conjugate (= transpose)** of the usual `R = x xᴴ`. If your method expects `R = x xᴴ`, use `np.conj(S)`.

**Length.** Audio is cut into 100 ms frames. Any trailing audio shorter than 100 ms is dropped.

## 5. Notes on the other scripts in this repo

These scripts are older and do **not** match the released checkpoint. Use `inference_cdbpn.py` instead:

- `models4ch/analyze_upsamp_mat.py` comes from an earlier experiment:
  - it uses 15 linearly spaced bands (50–9000 Hz) and an older copy of the model file (`DBPN.py`, the same architecture as `cdbpn.py`);
  - it will not load `cdbpn_epoch_99.pth`, because that checkpoint expects 16 bands.
- `cdbpn_upsample/train_cdbpn.py` still has `num_channels=9` from an earlier 9-band run. The released checkpoint was trained with `num_channels=16` and `scale_factor=8`.
- The setup that matches the checkpoint is the one in `models4ch/cls_feature_class.py` (`_get_visibility_matrix_for_file`) and `models4ch/seldnet_model.py`. `inference_cdbpn.py` copies it.
