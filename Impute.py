import pandas as pd

instances = pd.read_csv('instances.csv', delimiter='\t')

# get median of Environmental Audio Exposure (dBASPL)
medianAudio = instances['Environmental Audio Exposure (dBASPL)'].median()

# ge median of Blood Oxygen Saturation (%)
medianBlood = instances['Blood Oxygen Saturation (%)'].median()

print('Median of Environmental Audio Exposure (dBASPL):', medianAudio)
print('Median of Blood Oxygen Saturation (%):', medianBlood)

instances['Environmental Audio Exposure (dBASPL)'].fillna(medianAudio, inplace=True)
instances['Blood Oxygen Saturation (%)'].fillna(medianBlood, inplace=True)
instances['Walking Speed (km/hr)'].fillna(0.0, inplace=True)

instances.to_csv('instances_imputed.csv', sep='\t', index=False)
