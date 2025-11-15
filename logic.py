import numpy as np

def find_hubs(power_array, threshold=0.75):
    """
    Finds the primary hub and connected spokes for each time point.

    Args:
        power_array (np.ndarray): 2D array of (n_channels, n_times).
        threshold (float): A value between 0 and 1. A channel is a "spoke"
                           if its power is > (threshold * max_power).

    Returns:
        list: A list of dictionaries matching the data contract.
    """
    hub_results = []
    
    # We transpose the array to easily loop through time points
    for time_slice in power_array.T:
        # Find the channel with the maximum power for this time slice
        max_power = np.max(time_slice)
        hub_index = int(np.argmax(time_slice)) 

        # Find all channels that exceed the power threshold
        # These are our "spokes"
        spoke_indices, = np.where(time_slice > threshold * max_power)

        # Convert to a standard Python list and remove the hub itself
        spoke_indices_list = [int(i) for i in spoke_indices if i != hub_index]

        # Append the result for this time frame to our list
        hub_results.append({
            'hub': hub_index,
            'spokes': spoke_indices_list
        })
        
    return hub_results
