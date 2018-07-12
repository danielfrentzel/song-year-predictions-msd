"""
Loads Million Song Subset files from h5_files.pkl, places selected fields into a dictionary, sorts through the 
data and creates song_data.pkl, song_data_trimmed.pkl, song_data_timbre.pkl, song_data_timbre_pitch.pkl
"""


from collections import defaultdict
import hdf5_getters as hdf5_getters
import math
import numpy as np
import pandas as pd
import sys
import time


start_time = time.time()

# get all tracks from pickle
h5_files = pd.read_pickle('./pkl/h5_files.pkl')


def get_fields(files):
    tracks = []
    counts = {}
    field_counts = []
    for file in files:
        h5 = hdf5_getters.open_h5_file_read(file)
        t = {}
        t['artist_familiarity'] = hdf5_getters.get_artist_familiarity(h5)  # estimation
        t['artist_hotttnesss'] = hdf5_getters.get_artist_hotttnesss(h5)  # estimation
        t['artist_name'] = hdf5_getters.get_artist_name(h5)  # artist name
        t['release'] = hdf5_getters.get_release(h5)  # album name
        t['title'] = hdf5_getters.get_title(h5)  # title
        t['len_similar_artists'] = len(hdf5_getters.get_similar_artists(h5))  # number of similar artists
        t['analysis_sample_rate'] = hdf5_getters.get_analysis_sample_rate(h5)  # sample rate of the audio used ?????????
        t['duration'] = hdf5_getters.get_duration(h5)  # seconds
        t['key'] = hdf5_getters.get_key(h5)  # key the song is in
        t['key_confidence'] = hdf5_getters.get_key_confidence(h5)  # confidence measure
        t['loudness'] = hdf5_getters.get_loudness(h5)  # overall loudness in dB
        t['mode_confidence'] = hdf5_getters.get_mode_confidence(h5)  # confidence measure
        t['start_of_fade_out'] = hdf5_getters.get_start_of_fade_out(h5)  # time in sec
        t['tempo'] = hdf5_getters.get_tempo(h5)  # estimated tempo in BPM
        t['time_signature'] = hdf5_getters.get_time_signature(h5)  # estimate of number of beats per bar, e.g. 4
        t['year'] = hdf5_getters.get_year(h5)  # song release year from MusicBrainz or 0

        timbre = hdf5_getters.get_segments_timbre(h5)  # 2D float array, texture features (MFCC+PCA-like)
        t['segments_timbre'] = timbre
        t['timbre_avg'] = timbre.mean(axis=0)  # list of 12 averages
        cov_mat_timbre = np.cov(timbre, rowvar=False)
        cov_timbre = []
        for i in range(len(cov_mat_timbre)):
            for j in range(len(cov_mat_timbre) - i):
                cov_timbre.append(cov_mat_timbre[i][j])
        t['timbre_cov'] = cov_timbre  # list of 78 covariances

        pitch = hdf5_getters.get_segments_pitches(h5)  # 2D float array, chroma feature, one value per note
        t['segments_pitch'] = pitch
        t['pitch_avg'] = pitch.mean(axis=0)  # list of 12 averages
        cov_mat_pitch = np.cov(pitch, rowvar=False)
        cov_pitch = []
        for i in range(len(cov_mat_pitch)):
            for j in range(len(cov_mat_pitch) - i):
                cov_pitch.append(cov_mat_timbre[i][j])
        t['pitch_cov'] = cov_pitch  # list of 78 covariances

        # seg_pitch = hdf5_getters.get_segments_pitches(h5)  # 2D float array, chroma feature, one value per note
        # print(seg_pitch.shape)

        # t['artist_latitude'] = hdf5_getters.get_artist_latitude(h5)  # float, ????????????????????????????????????????
        # t['artist_longitude'] = hdf5_getters.get_artist_longitude(h5)  # float, ??????????????????????????????????????
        # t['artist_location'] = hdf5_getters.get_artist_location(h5)  # location name
        # t['song_hotttnesss'] = hdf5_getters.get_song_hotttnesss(h5)  # estimation
        # t['danceability'] = hdf5_getters.get_danceability(h5)  # estimation
        # t['end_of_fade_in'] = hdf5_getters.get_end_of_fade_in(h5)  # seconds at the beginning of the song
        # t['energy'] = hdf5_getters.get_energy(h5)  # energy from listener point of view
        # t['mode'] = hdf5_getters.get_mode(h5)  # major or minor
        # t['time_signature_confidence'] = hdf5_getters.get_time_signature_confidence(h5)  # confidence measure
        # t['artist_mbtags_count'] = len(hdf5_getters.get_artist_mbtags_count(h5))  # array int, tag counts for musicbrainz tags
        # bad types or non arithmatic numbers
        '''
        # t['audio_md5'] = hdf5_getters.get_audio_md5(h5)  # hash code of the audio used for the analysis by The Echo Nest
        # t['artist_terms_weight'] = hdf5_getters.get_artist_terms_weight(h5)  # array float, echonest tags weight ?????
        # t['artist_terms_freq'] = hdf5_getters.get_artist_terms_freq(h5)  # array float, echonest tags freqs ??????????
        # t['artist_terms'] = hdf5_getters.get_artist_terms(h5)  # array string, echonest tags ?????????????????????????
        # t['artist_id'] = hdf5_getters.get_artist_id(h5)  # echonest id
        # t['artist_mbid'] = hdf5_getters.get_artist_mbid(h5)  # musicbrainz id
        # t['artist_playmeid'] = hdf5_getters.get_artist_playmeid(h5)  # playme id
        # t['artist_7digitalid'] = hdf5_getters.get_artist_7digitalid(h5)  # 7digital id
        # t['release_7digitalid'] = hdf5_getters.get_release_7digitalid(h5)  # 7digital id
        # t['song_id'] = hdf5_getters.get_song_id(h5)  # echonest id
        # t['track_7digitalid'] = hdf5_getters.get_track_7digitalid(h5)  # 7digital id
        # t['similar_artists'] = hdf5_getters.get_similar_artists(h5)  # string array of sim artist ids
        # t['track_id'] = hdf5_getters.get_track_id(h5)  # echonest track id
        # t['segments_start'] = hdf5_getters.get_segments_start(h5)  # array floats, musical events, ~ note onsets
        # t['segments_confidence'] = hdf5_getters.get_segments_confidence(h5)  # array floats, confidence measure
        # t['segments_pitches'] = hdf5_getters.get_segments_pitches(h5)  # 2D float array, chroma feature, one value per note
        # t['segments_timbre'] = hdf5_getters.get_segments_timbre(h5)  # 2D float array, texture features (MFCC+PCA-like)
        # t['segments_loudness_max'] = hdf5_getters.get_segments_loudness_max(h5)  # float array, max dB value
        # t['segments_loudness_max_time'] = hdf5_getters.get_segments_loudness_max_time(h5)  # float array, time of max dB value, i.e. end of attack
        # t['segments_loudness_start'] = hdf5_getters.get_segments_loudness_start(h5)  # array float, dB value at onset
        # t['sections_start'] = hdf5_getters.get_sections_start(h5)  # array float, largest grouping in a song, e.g. verse
        # t['sections_confidence'] = hdf5_getters.get_sections_confidence(h5)  # array float, confidence measure
        # t['beats_start'] = hdf5_getters.get_beats_start(h5)  # array float, result of beat tracking
        # t['beats_confidence'] = hdf5_getters.get_beats_confidence(h5)  # array float, confidence measure
        # t['bars_start'] = hdf5_getters.get_bars_start(h5)  # array float, beginning of bars, usually on a beat
        # t['bars_confidence'] = hdf5_getters.get_bars_confidence(h5)  # array float, confidence measure
        # t['tatums_start'] = hdf5_getters.get_tatums_start(h5)  # array float, smallest rythmic element
        # t['tatums_confidence'] = hdf5_getters.get_tatums_confidence(h5)  # array float, confidence measure
        # t['artist_mbtags'] = hdf5_getters.get_artist_mbtags(h5)  # array string, tags from musicbrainz.org 
        '''
        h5.close()

        for key, value in t.items():
            if isinstance(value, float) and math.isnan(value):
                pass
            if type(value) is np.ndarray:
                if key in counts.keys():
                    counts[key] += 1
                else:
                    counts[key] = 1
            elif value:
                if key in counts.keys():
                    counts[key] += 1
                else:
                    counts[key] = 1
            elif key not in counts.keys():
                counts[key] = 0

        count = 0
        for key, value in t.items():
            if isinstance(value, float) and math.isnan(value):
                pass
            elif type(value) is np.ndarray:
                count += 1
            elif value:
                count += 1
        field_counts.append(count)

        # progress bar
        if num_of_tracks >= 100:
            i = files.index(file) + 1
            scale = num_of_tracks/100
            if i % math.ceil(len(files)*.05) == 0:
                sys.stdout.write('\r')
                # the exact output you're looking for:
                sys.stdout.write("Loading dataframe: [%-100s] %d%%" % ('=' * int(i//scale), 1/scale * i))
                sys.stdout.flush()
                time.sleep(.01)

        tracks.append(t)
    print()
    return tracks, counts, field_counts


num_of_tracks = 10000 # if you don't want to test with all 10,000 tracks
tracks, counts, field_counts = get_fields(h5_files[:num_of_tracks])

# count percentages of tracks that have particular field
for key, value in sorted(counts.items()):
    print(key, str(100*value/num_of_tracks)+'%')

# count tracks that have all fields
all_fields = 0
for c in field_counts:
    if c == len(counts.keys()):
        all_fields += 1
print(str(all_fields/num_of_tracks*100) + '%', 'of songs have all fields')

# get all tracks
track_info = defaultdict(list)
for t in tracks:
    for key, value in t.items():
        track_info[key].append(value)

# get only tracks that have all fields
track_info_trimmed = defaultdict(list)
for i in range(len(tracks)):
    if field_counts[i] == len(counts.keys()):
        for key, value in sorted(tracks[i].items()):
            track_info_trimmed[key].append(value)

# get only tracks that have year and timbre
track_info_timbre = defaultdict(list)
for i in range(len(tracks)):
    t = tracks[i]
    if t['year'] > 0 and len(t['timbre_avg']) == 12 and len(t['timbre_cov']) == 78:
        track_info_timbre['year'].append(t['year'])
        for j in range(len(t['timbre_avg'])):
            key = 'timbre_avg' + str(j + 1)
            track_info_timbre[key].append(t['timbre_avg'][j])
        for k in range(len(t['timbre_cov'])):
            key = 'timbre_cov' + str(k + 13)
            track_info_timbre[key].append(t['timbre_cov'][k])

track_info_timbre_pitch = defaultdict(list)
for i in range(len(tracks)):
    t = tracks[i]
    if t['year'] > 0 and len(t['timbre_avg']) == 12 and len(t['timbre_cov']) == 78 and len(t['pitch_avg']) == 12 and len(t['pitch_cov']) == 78:
        track_info_timbre_pitch['year'].append(t['year'])
        for j in range(len(t['timbre_avg'])):
            key = 'timbre_avg' + str(j + 1)
            if j < 9:
                key = 'timbre_avg0' + str(j + 1)
            track_info_timbre_pitch[key].append(t['timbre_avg'][j])
        for k in range(len(t['timbre_cov'])):
            key = 'timbre_cov' + str(k + 13)
            track_info_timbre_pitch[key].append(t['timbre_cov'][k])
        for l in range(len(t['pitch_avg'])):
            key = 'pitch_avg' + str(l + 1)
            if l < 9:
                key = 'pitch_avg0' + str(l + 1)
            track_info_timbre_pitch[key].append(t['pitch_avg'][l])
        for m in range(len(t['pitch_cov'])):
            key = 'pitch_cov' + str(m + 13)
            track_info_timbre_pitch[key].append(t['pitch_cov'][m])


df = pd.DataFrame(track_info)

df.to_pickle('./pkl/song_data.pkl')

# pickle only tracks that have all fields
df_trimmed = pd.DataFrame(track_info_trimmed)

df_trimmed.to_pickle('./pkl/song_data_trimmed.pkl')

# pickle tracks with year and timbre
pd.DataFrame(track_info_timbre).to_pickle('./pkl/song_data_timbre.pkl')
pd.DataFrame(track_info_timbre_pitch).to_pickle('./pkl/song_data_timbre_pitch.pkl')

# print(sorted(track_info_timbre_pitch.keys()))
# print(sorted(track_info_timbre.keys()))

print("--- %s seconds ---" % (time.time() - start_time))

# DELETE
# df_timbre_pitch = pd.read_pickle('./pkl/song_data_timbre_pitch.pkl')
# print(len(df_timbre_pitch))
#
# df_timbre = pd.read_pickle('./pkl/song_data_timbre.pkl')
# print(len(df_timbre))