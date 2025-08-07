# Import necessary libraries
import os
import mne
import pyxdf
import numpy as np

# --- Configuration ---
# ������ Please replace this with the actual path to your .xdf file ������
fname = r"C:\Users\otsuki\Documents\CurrentStudy\sub-P002\ses-S001\eeg\sub-P002_ses-S001_task-Default_run-001_eeg.xdf"

# --- Main Logic ---
if not os.path.exists(fname):
    print(f"Error: File not found at the specified path: {fname}")
else:
    try:
        # 1. Load the entire XDF file to find all streams
        streams, header = pyxdf.load_xdf(fname)
        print("Successfully loaded the XDF file.")

        # 2. Find the main physiological data stream (assuming it's the first 'EEG' type)
        eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
        if eeg_stream is None:
            raise RuntimeError("Could not find any stream with type 'EEG' in the file.")

        # 3. Extract data and metadata from the EEG stream
        data = np.array(eeg_stream['time_series']).T  # Transpose to (n_channels, n_times)
        sfreq = float(eeg_stream['info']['nominal_srate'][0])
        ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
        
        # 4. Dynamically set channel types based on their names
        ch_types = []
        for name in ch_names:
            name_upper = name.upper()
            if 'EOG' in name_upper:
                ch_types.append('eog')
            elif 'TRIGGER' in name_upper:
                ch_types.append('stim') # 'stim' is the correct type for trigger channels
            else:
                ch_types.append('eeg')
        
        print(f"\nIdentified {len(ch_names)} channels. Setting types automatically.")
        # print(list(zip(ch_names, ch_types))) # Uncomment to verify channel types

        # 5. Create the MNE Info object and the RawArray object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        
        # Set a standard montage for the EEG channels for later plotting and source localization
        # This will ignore (only for visualization, since 10-20 layout doesn't have placde for EOG) non-EEG channels like 'EOG' and 'TRIGGER'
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn') # 'warn' will tell us if any EEG channels don't match

        print("Created MNE Raw object.")
        
        # 6. Find and process the marker stream
        marker_stream = next((s for s in streams if 'Marker' in s['info']['type'][0]), None)
        if marker_stream:
            print(f"Found marker stream: '{marker_stream['info']['name'][0]}'")
            
            # Extract marker timings and descriptions
            marker_onsets = marker_stream['time_stamps'] - eeg_stream['time_stamps'][0] # Align to EEG start time
            marker_descriptions = [str(label[0]) for label in marker_stream['time_series']]
            marker_durations = np.zeros(len(marker_onsets)) # Markers are events, so duration is 0

            # Create an Annotations object
            annotations = mne.Annotations(onset=marker_onsets,
                                          duration=marker_durations,
                                          description=marker_descriptions,
                                          orig_time=raw.info.get('meas_date')) # Synchronize with raw's measurement date
            
            # Add the annotations to the Raw object
            raw.set_annotations(annotations)
            print("Successfully added markers as annotations.")
        else:
            print("Warning: No marker stream found in the file.")

        # 7. Visualize the final, combined data
        print("\nPlotting the data. Close the plot window to continue...")
        # The plot will show all channels, and annotations will appear as vertical lines
        raw.plot(scalings='auto', n_channels=15, show_scrollbars=True, block=True)
        
        print("\nScript finished. You can now proceed with preprocessing.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
