The sampling is somewhat difficult. The solution I settled on is the following:

for our curriculum sampling, set the difficulty in the CurriculumInfoWrapper
and transmit difficulties to the CurriculumEnvLog which sample and resample
themselves.

