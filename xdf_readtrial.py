# Import necessary libraries
import os
import mne
import pyxdf
import numpy as np

# --- Configuration ---
# ↓↓↓↓↓↓ Please replace this with the actual path to your .xdf file ↓↓↓↓↓↓
fname = r"C:\Users\otsuki\Documents\CurrentStudy\sub-P002\ses-S001\eeg\sub-P002_ses-S001_task-Default_run-001_eeg.xdf"

# --- Main Logic ---
if not os.path.exists(fname):
    print(f"Error: File not found at the specified path: {fname}")
else:
    try:
        # 1. Load the entire XDF file to find all streams
        streams, header = pyxdf.load_xdf(fname)
        print("Successfully loaded the XDF file.")

        # 2. Display all available streams to the user
        print("\n--- Available Streams ---")
        if not streams:
            raise RuntimeError("No streams found in the XDF file.")
        
        for i, stream in enumerate(streams):
            stream_name = stream['info']['name'][0]
            stream_type = stream['info']['type'][0]
            print(f"  [{i}]: Name: '{stream_name}', Type: '{stream_type}'")
        print("-------------------------\n")

        # 3. Ask user to select the EEG stream
        while True:
            try:
                eeg_choice_idx = int(input("Enter the number of the EEG stream to process: "))
                if 0 <= eeg_choice_idx < len(streams):
                    eeg_stream = streams[eeg_choice_idx]
                    print(f"Selected EEG stream: '{eeg_stream['info']['name'][0]}'")
                    break
                else:
                    print("Invalid number. Please choose from the list above.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        # 4. Ask user to select the Marker stream (with an option to skip)
        while True:
            try:
                marker_choice_str = input("Enter the number of the Marker stream to use (or type 'none' to skip): ")
                if marker_choice_str.lower() == 'none':
                    marker_stream = None
                    print("Skipping marker stream processing.")
                    break
                
                marker_choice_idx = int(marker_choice_str)
                if 0 <= marker_choice_idx < len(streams):
                    marker_stream = streams[marker_choice_idx]
                    print(f"Selected marker stream: '{marker_stream['info']['name'][0]}'")
                    break
                else:
                    print("Invalid number. Please choose from the list above.")
            except ValueError:
                print("Invalid input. Please enter a number or 'none'.")

        # 5. Extract data and metadata from the selected EEG stream
        data = np.array(eeg_stream['time_series']).T  # Transpose to (n_channels, n_times)
        sfreq = float(eeg_stream['info']['nominal_srate'][0])
        ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
        
        # 6. Dynamically set channel types based on their names
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
        
        # 7. Create the MNE Info object and the RawArray object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn')

        print("Created MNE Raw object.")
        
        # 8. Process the selected marker stream (if one was chosen)
        if marker_stream:
            print(f"Processing selected marker stream...")
            
            marker_onsets = marker_stream['time_stamps'] - eeg_stream['time_stamps'][0]
            marker_descriptions = [str(label[0]) for label in marker_stream['time_series']]
            marker_durations = np.zeros(len(marker_onsets))

            annotations = mne.Annotations(onset=marker_onsets,
                                          duration=marker_durations,
                                          description=marker_descriptions,
                                          orig_time=raw.info.get('meas_date'))
            
            raw.set_annotations(annotations)
            print("Successfully added markers as annotations.")
        else:
            print("No marker stream was selected.")

        # 9. Visualize the final, combined data
        print("\nPlotting the data. Close the plot window to continue...")
        raw.plot(scalings='auto', n_channels=15, show_scrollbars=True, block=True)
        
        print("\nScript finished. You can now proceed with preprocessing.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
