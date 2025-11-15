"""
The data will be a list of dictionaries.
Each dictionary represents one time-step or animation frame.

Structure of each dictionary:
{
  'hub': An integer representing the index of the hub sensor.
  'spokes': A list of integers for the spoke sensor indices.
}

Example data for 3 animation frames:
"""
data_for_visualization = [
		{
				'hub': 10, 
				'spokes': [5, 15, 20] 
		},
		{
				'hub': 12, 
				'spokes': [8, 11, 14, 25]
		},
		{
				'hub': 31, 
				'spokes': [28, 30, 35, 40, 55] 
		},
]
