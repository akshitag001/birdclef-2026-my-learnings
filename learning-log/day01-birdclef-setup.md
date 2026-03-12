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

   Result shape : 128 x 313 ( meaning 128 frequency bands , 313 time frames )

4. Built a PyTorch Dataset

 i create a custom dataset class purpose was to load data than convert to spectogram than attach label   
 each training sample contains ( spectogram tensor + special label ) labels were encoded into a 234 length vector. 
 
 what i learned : MULTI LABEL CLASSIFICATION

5. Create a DataLoader

   we used a pytorch dataloader to efficiently feed data to the model , ex batch shape [16,1,128,313] coz , train model faster using batches

6. MAIN PART ( BUILT THE NEURAL NETWORK )

   we use ResNet18, a convolutional neural network , coz its fast stable baseline , good for image like data

   since spectograms are grayscale , we modified the first layer:
   
   input channels : 3 -> 1  ( the final layer outputs will be 234 species probabilities )

8. Defined Loss Function

   we use BCEWithLogitsLoss , coz it is a multi label classification problem , one audio can contain 2-3 species voice mix.

9. Train the Model

 spectrogram
   
   ↓
   
   model prediction
   
   ↓
   
   compare with true label
   
   ↓
   
   calculate loss
   
   ↓
   
   update weights
   
   i train the model for 3 epochs 
   <img width="612" height="362" alt="image" src="https://github.com/user-attachments/assets/95a1636a-1d75-41c7-8a9f-834ec8bedded" />

   than i save the trained Model with its weights: birdclef_resnet18.pth  
   
   purpose to reuse trained model for prediction without retraining. 

