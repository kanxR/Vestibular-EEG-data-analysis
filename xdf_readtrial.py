import pyxdf
import mne
import numpy as np
import os

fname = r'your file path'

if not os.path.exists(fname):
    print(f"Error: File not found: {fname}")
else:
    streams, header = pyxdf.load_xdf(fname)

    # --- Find EEG stream ---
    eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
    if eeg_stream is None:
        print("No EEG stream found.")
        exit()

    # --- Extract EEG data and metadata ---
    data = np.array(eeg_stream['time_series']).T
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
    n_channels = len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * n_channels)

    # --- Create Raw object ---
    raw = mne.io.RawArray(data, info)

    # --- Print EEG metadata ---
    print("EEG Metadata:")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Number of channels: {n_channels}")
    print(f"  Channel names: {ch_names}")
    print(f"  Data shape: {data.shape}")

    # --- Find marker streams (any stream with 'int' in format) ---
    marker_streams = [s for s in streams if 'int' in s['info']['channel_format'][0].lower()]
    if marker_streams:
        print(f"Found {len(marker_streams)} marker stream(s): {[s['info']['name'][0] for s in marker_streams]}")
        eeg_start = eeg_stream['time_stamps'][0]
        all_annotations = []

        for marker_stream in marker_streams:
            marker_onsets = np.array(marker_stream['time_stamps']) - eeg_start
            marker_labels = [str(ts[0]) for ts in marker_stream['time_series']]
            durations = np.zeros(len(marker_onsets))
            orig_time = raw.info.get('meas_date', None)
            ann = mne.Annotations(onset=marker_onsets, duration=durations, description=marker_labels, orig_time=orig_time)
            all_annotations.append(ann)

        # Combine all annotations
        combined_annotations = all_annotations[0]
        for ann in all_annotations[1:]:
            combined_annotations += ann

        raw.set_annotations(combined_annotations)
        print("All marker annotations added to Raw object.")
    else:
        print("No marker streams found.")

    # --- Visualize EEG with markers ---
    print("Plotting EEG data with markers...")
    raw.plot(scalings='auto', n_channels=min(15, n_channels), show_scrollbars=True, block=True)