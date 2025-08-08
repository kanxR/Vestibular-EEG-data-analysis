# Import necessary libraries
import os
import mne
from copy import deepcopy # Import deepcopy as per the documentation
import matplotlib.pyplot as plt

def main():
    """
    Main function to run the bad channel detection and interpolation pipeline.
    """
    # --- Configuration ---
    # ★★★ Define the input and output file paths ★★★
    base_dir = r'C:\Users\otsuki\Documents\CurrentStudy\sub-P002\ses-S001\eeg'
    subject_id = 'sub-P002_ses-S001_task-Default_run-001'
    
    input_fname = os.path.join(base_dir, f"{subject_id}_filtered_raw.fif")
    output_fname = os.path.join(base_dir, f"{subject_id}_interpolated_raw.fif")

    # --- Step 1: Load the Filtered Data ---
    if not os.path.exists(input_fname):
        print(f"Error: Input file not found: {input_fname}")
        print("Please make sure you have run the '01_Load_and_Filter.py' script first.")
        return
        
    print(f"Loading filtered data from: {input_fname}")
    raw = mne.io.read_raw(input_fname, preload=True)

    # --- Step 2: Visual Inspection for Bad Channels ---
    # This script now follows the manual inspection method from the documentation.
    print("\n--- Step 2: Visual Inspection ---")
    print("The EEG data will now be plotted. Please inspect the channels.")
    print("What to look for:")
    print("  - Channels that are completely flat (no signal).")
    print("  - Channels that are excessively noisy and look very different from their neighbors.")
    print("\nACTION: In the plot window, LEFT-CLICK on the name of a channel to mark it as 'bad'.")
    print("        You can click it again to unmark it.")
    print("        When you are finished, CLOSE the plot window.")
    
    # The 'bads' list is initially empty.
    raw.info['bads'] = []
    
    # Plot the data. This is an interactive plot.
    raw.plot(scalings='auto', n_channels=20, show_scrollbars=True, block=True)

    # The channels you clicked on are now stored in raw.info['bads']
    print(f"\nChannels you marked as bad: {raw.info['bads']}")

    # --- Step 3: Confirm and Finalize Bad Channels ---
    print("\n--- Step 3: Finalize Bad Channels ---")
    
    # Use deepcopy to safely handle the list of bads
    final_bads = deepcopy(raw.info['bads'])
    
    while True:
        print("\nYour current list of bad channels is:", final_bads)
        user_input = input("Press ENTER to confirm. Or, type a channel name to add/remove it (e.g., 'Fp1'), then press ENTER: ")
        
        if user_input == "":
            print("List confirmed.")
            break
        else:
            channel_to_toggle = user_input.strip().upper()
            
            if channel_to_toggle not in [ch.upper() for ch in raw.ch_names]:
                 print(f"Error: Channel '{channel_to_toggle}' not found. Please try again.")
                 continue

            actual_channel_name = next(ch for ch in raw.ch_names if ch.upper() == channel_to_toggle)

            if actual_channel_name in final_bads:
                final_bads.remove(actual_channel_name)
                print(f"Removed '{actual_channel_name}'.")
            else:
                final_bads.append(actual_channel_name)
                print(f"Added '{actual_channel_name}'.")

    raw.info['bads'] = final_bads
    print(f"\nFinal list of bad channels to be interpolated: {raw.info['bads']}")

    # --- Step 4: Interpolate Bad Channels ---
    if not raw.info['bads']:
        print("\nNo bad channels were selected. Skipping interpolation.")
        raw_interpolated = raw.copy()
    else:
        print("\n--- Step 4: Interpolating Bad Channels ---")
        raw_interpolated = raw.copy()
        raw_interpolated.interpolate_bads(reset_bads=True, mode='accurate')
        print("Interpolation complete.")

        print("\nPlotting the data again to show the interpolated channels (in red).")
        raw_interpolated.plot(scalings='auto', n_channels=20, show_scrollbars=True, block=True)

    # --- Step 5: Save the Cleaned Data ---
    print(f"\n--- Step 5: Saving Data ---")
    print(f"Saving data with interpolated channels to: {output_fname}")
    raw_interpolated.save(output_fname, overwrite=True)
    print("Done. You can now use this .fif file for the next step.")

if __name__ == "__main__":
    main()
