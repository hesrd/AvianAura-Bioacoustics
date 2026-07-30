Drop your primary-data recordings into the matching sub-folder:
  PrimaryData/urban/song/*.mp3 (or .wav / .flac / .m4a)
  PrimaryData/urban/call/
  PrimaryData/rural/song/
  PrimaryData/rural/call/

They will be preprocessed with IDENTICAL settings to the
Xeno-canto data (500 Hz Butterworth high-pass, RMS normalisation,
same STFT and Mel settings) so the two sources are directly
comparable in the combined analysis.

Each primary spectrogram is written with a 'PRIMARY_' filename
prefix so it can be identified in the manifest and downstream
analysis if you want to count how many were personally recorded.
