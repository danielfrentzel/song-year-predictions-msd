""" 
Loads the song data from song_data_timbre_pitch.pkl into a Comma Seperated Value formated file
"""

import csv
import pandas as pd

df = pd.read_pickle('song_data_timbre_pitch.pkl')

mega = []
for k in list(df.keys()):
    mega.append(list(df[k]))

t = []
for i in range(len(mega[0])):
    t1 = []
    for l in mega:
        t1.append(l[i])
    t.append(t1)

# print(len(t), len(t[0]))
# print(len(mega), len(mega[0]))

with open('msd.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(df.keys())
    for r in t:
        w.writerow(r)
