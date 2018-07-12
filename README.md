# song-year-predictions-msd
Using the Million Song Dataset Subset to predict song year based on sound characteristics, namely timbre and pitch.
See full paper: https://danielfrentzel.github.io/static/MLSSpaper.pdf

Immediatley after cloning the repository, data_analysis.py can be ran to recreate the .png graphs based on the existing .pkl files.

***In order recreate all .pkl files by running the python files, the Million Song Subset needs to be downloaded and placed into the song-year-predictions-msd directory. https://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset#subset

If MillionSongSubset has been downloaded an placed in song-year-predictions-msd, The following python files can be ran in order to recreate all project files: get_h5_files.py, load_dataframe.py, data_analysis.py, msd_csv.py

Python file descriptions:</br>
get_h5_files.py - Collects all h5 files from the MillionSongSubset folder and places them into pkl/h5_files.pkl</br>
load_dataframe.py - Loads files from h5_files.pkl and creates various .pkl files for data analysis (around 10 minute runtime)</br>
data_analysis.py - Uses .pkl files to run leave-one-out cross-validation ridge regression and creates .png files displaying a histogram of song year prediction accuracy and a graph of variable coefficients due to regulation. </br>
msd_csv.py - Loads the song data from song_data_timbre_pitch.pkl into a Comma Seperated Value formated file</br>
hdf5_getters.py - included file from MSD creators, used by load_dataframe.py</br>

Blog: https://danielfrentzel.github.io/blog2/
