# Import necessary libraries
import pyxdf
import os

# --- Configuration ---
# ÅöÅöÅö Please replace this with the actual path to your .xdf file ÅöÅöÅö
fname = r'your_file_path_here.xdf'

# --- Main Logic ---
# Check if the file exists before trying to load it
if not os.path.exists(fname):
    print(f"Error: File not found at the specified path: {fname}")
else:
    try:
        # Load the entire XDF file to ensure all streams are found.
        # load_xdf() is more reliable than resolve_streams() for complex files.
        streams, header = pyxdf.load_xdf(fname)

        # Print a header for the investigation report
        print(f"Investigating contents of '{os.path.basename(fname)}'...")
        print("-" * 60)

        # Loop through all found streams and print their detailed information
        if not streams:
            print("No streams found in the file.")
        else:
            for i, stream in enumerate(streams):
                # Safely get stream info using .get() to avoid errors if a key is missing
                info = stream.get('info', {})
                name = info.get('name', ['N/A'])[0]
                stype = info.get('type', ['N/A'])[0]
                n_channels = int(info.get('channel_count', [0])[0])

                # Print the basic information for the current stream
                print(f"\n[Stream {i+1}]")
                print(f"  Name: '{name}'")
                print(f"  Type: '{stype}'")
                print(f"  Number of Channels: {n_channels}")

                # Try to find and display detailed channel labels if they exist
                try:
                    # Channel information is often nested within the 'desc' dictionary
                    channels_info = info.get('desc', [{}])[0].get('channels', [{}])[0].get('channel', [])
                    if channels_info:
                        print(f"  Channel List:")
                        # Extract the 'label' for each channel
                        ch_names = [ch.get('label', ['N/A'])[0] for ch in channels_info]
                        for j, ch_name in enumerate(ch_names):
                            print(f"    {j+1}: '{ch_name}'")
                    else:
                        print("  Detailed channel labels not found in the 'desc' section.")
                except (KeyError, IndexError):
                    # This handles cases where the 'desc' structure is different or missing
                    print("  Could not parse detailed channel labels.")

        # Print a footer for the report
        print("\n" + "-" * 60)
        print("Investigation complete.")

    except Exception as e:
        # Catch any other errors that might occur during file loading or processing
        print(f"An unexpected error occurred: {e}")
