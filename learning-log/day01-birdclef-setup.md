# Day 1 - BirdCLEF 2026 Setup
Date : March 12, 2026
First , i will show you the result or leaderboard then we will discuss how i got it .....

## RESULT

<img width="834" height="348" alt="image" src="https://github.com/user-attachments/assets/b35efa2f-1ae8-4fa4-8ef7-3b0e67cc9d33" />

## Detail
1. Loaded the dataset from kaggle in the kaggle notebook only , have list of 234 species classes
   Understand how audio data and species labels are strucured.
2. Explored Dataset Structure
   train_audio/1161364/iNat1216197.ogg
   yeah so got to know that mp3 is not only the audio format , .ogg is also an audio format it uses the Ogg container , mostly it contains audio compressed with the Ogg Vorbis codec.

3. Converted Audio ---> Spectogram
   coz, neural network cannot directly understand raw audio , so we converted audio into Mel Spectrograms using Librosa ( yeah the same library we use in ai voice detection project )
audio wavefrom ----> mel spectogram ---> image like representations
ex. code
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=32000,
    n_mels=128
)
   
