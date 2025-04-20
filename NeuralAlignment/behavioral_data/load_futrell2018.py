"""
Inspect the Futrell2018 behavioral data. It can be downloaded from brain-score language and loaded using a brain-score
environment containing xarray.
https://github.com/brain-score/language
"""

import xarray as xr
import numpy as np

# Load 'assy_Futrell2018.nc'
data = xr.open_dataset('assy_Futrell2018.nc')

"""
n=179 if excluding one participant with too few data points
we only included data for stories where they answered 5 or all 6 comprehension questions
we excluded reading times (RTs) that were shorter than 100 ms or longer than 3000 ms (per Futrell et al., 2018 (and Schrimpf et al., 2021))
"""

assert (np.unique(data.correct.values) == [5, 6]).all() # Each story was accompanied by 6 comprehension questions, where participants chose the correct answer from a set of two

response_data = data.data.values # words by participants, should be in milliseconds. The associated words are in data.word.values
assert (np.nanmin(response_data) >= 100) & (np.nanmax(response_data) <= 3000)

# count how many samples each participant has
n_samples = np.sum(~np.isnan(response_data), axis=0) # last subject only has 6 trials, exclude

# story_id has the story and the words should be contextualized by the story
