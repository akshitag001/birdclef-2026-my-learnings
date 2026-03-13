## So today rather than running a notebook , i read the code of other top performers that what they are using , what new techniques in ML/DL used in their notebook.

one of the thing i got to know is OpenVino

The below code has used OpenVino lets discuss what it is:

<img width="777" height="542" alt="image" src="https://github.com/user-attachments/assets/b21d1f71-085a-486a-8945-2926a1bc6a69" />

So OpenVINO is a toolkit created by Intel to make our AI models run faster during inference.

Think like an AI performance engine that takes trained model and optimizes it so it runs much faster on hardware like CPU's or GPU's and specialized chips.

1. Lets First Understand the AI lifecycle:
   
   Every Ml system has two main stages that are Training ( tools used pytorch , tensorflow ) this stage is very heavy and usually done on GPU's

   second is Inference where the trained models makes prediciton, this stage needs to be fast and efficient

2. The problem our this new tech OpenVINO solves :

   models which are trained with frameworks like pytorch are not optimized for production speed like for ex when its running on cpu , its prediction time can be lets take 200ms but same will give in 20ms in OpenVINO models.
   
   which is 10X faster , coz training frameworks prioritize flexibility , not executing speed.

4. What it actually done internally :
   
   It performs several optimizations like
   
   a. Model conversion ( .pth, .pb , .onnx ) into OpenVINO IR format ( .xml + .bin )
   
   b. Graph optumization ( conv -> batchnorm -> ReLU ) this fused into Optimized single operation
   
   c. Hardware optimization : it uses cpu vector instructions AVX , AVX2, AVX512 , these allow parallel math operation.
   
   d. Layer fusion : multiple layers are merged to redcue memory movement. ( biggest bottleneck in NN )
   
   e. Precision reduction ( FP32-FP16 ) or ( FP32->INT8) this speed up inference

4.How OpenVINO works step by step

  a. Train model
  
  b. Convert model
  
  c. Compile model
  
      core.compile_model(model, "CPU")
  d. Run inference
