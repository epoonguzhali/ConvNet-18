  
clc;
clear all;
close all;

% Augmentation using Rotation (Glaucoma images)
D='E:\poonguzhali\Rimone2Train\glaucoma'
S = dir(fullfile(D,'*.jpg')); % pattern to match filenames.
 i = 1;
for k = 1:numel(S)
k
    F = fullfile(D,S(k).name);
    In=imread(F);
    
     %
     [m,n,d] = size(In);
        for j = 0:1:4
        I1=imrotate(In,j);
          
        I1 = imresize(I1,[m,n]);

  imwrite(I1, strcat('E:\poonguzhali\Rimone2Training\glaucoma\',num2str(i,'%d'),'_rgl.jpg')); 
  
 i  =i+1;
        end
end 
cd('E:\poonguzhali\Rimone2Train\glaucoma');


D='E:\poonguzhali\Rimone2Test\glaucoma'
S = dir(fullfile(D,'*.jpg')); 
 i = 1;

for k = 1:numel(S)
k
    F = fullfile(D,S(k).name);
    In=imread(F);
    
     %
     [m,n,d] = size(In);
        for j = 0:1:4
        I1=imrotate(In,j);
          
        I1 = imresize(I1,[m,n]);

  imwrite(I1, strcat('E:\poonguzhali\Rimone2Testing\glaucoma\',num2str(i,'%d'),'_rgl.jpg')); 
  
 i  =i+1;
        end
end 
cd('E:\poonguzhali\Rimone2Test\glaucoma');

% Augmentation of Normal images

D='E:\poonguzhali\Rimone2Train\normal'
S = dir(fullfile(D,'*.jpg')); 
 i = 1;

for k = 1:numel(S)
k
    F = fullfile(D,S(k).name);
    In=imread(F);
    
     %
     [m,n,d] = size(In);
        for j = 0:1:1
        I1=imrotate(In,j);
          
        I1 = imresize(I1,[m,n]);
  imwrite(I1, strcat('E:\poonguzhali\Rimone2Training\normal\',num2str(i,'%d'),'_rgl.jpg')); 
  
 i  =i+1;
        end
end 
cd('E:\poonguzhali\Rimone2Train\normal');

D='E:\poonguzhali\Rimone2Test\normal'
S = dir(fullfile(D,'*.jpg')); 
 i = 1;
for k = 1:numel(S)
k
    F = fullfile(D,S(k).name);
    In=imread(F);
    
     %
     [m,n,d] = size(In);
        for j = 0:1:1
        I1=imrotate(In,j);
          
        I1 = imresize(I1,[m,n]);

  imwrite(I1, strcat('E:\poonguzhali\Rimone2Testing\normal\',num2str(i,'%d'),'_rgl.jpg')); 
  
 i  =i+1;
        end
end 
cd('E:\poonguzhali\Rimone2Test\normal');

%%
tic;

% Load images from Database

matlabroot = 'E:\poonguzhali'
Datasetpath = fullfile(matlabroot,'Rimone2Training')
FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

matlabroot = 'E:\poonguzhali'
Datasetpath = fullfile(matlabroot,'Rimone2Testing')
FinalTest  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

% Finding number of images in each category for training
labelCountTrain = countEachLabel(FinalTrain)

% Finding number of images in each category for testing
labelCountTest = countEachLabel(FinalTest)

% To make all the images(both training and testing) of equal size

inputSize = [64,64,3];

augimdsTrain = augmentedImageDatastore(inputSize(1:2),FinalTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),FinalTest);

% Constructing CNN Layers

layers = [imageInputLayer([64 64 3],'Name','input')
             convolution2dLayer(3,96,'Name','Conv1','WeightsInitializer','he','Padding','Same')
             batchNormalizationLayer('Name','BN1')
             reluLayer('Name','relu1')
             maxPooling2dLayer(2,'stride',2,'Name','max1')

             convolution2dLayer(3,96,'Name','Conv2','WeightsInitializer','he','Padding','Same')
             batchNormalizationLayer('Name','BN2')
             reluLayer('Name','relu2')
             maxPooling2dLayer(2,'stride',2,'Name','max2')
             
             convolution2dLayer(3,96,'Name','Conv3','WeightsInitializer','he','Padding','Same')
             batchNormalizationLayer('Name','BN3')
             reluLayer('Name','relu3')
                                    
             convolution2dLayer(3,96,'Name','Conv4','WeightsInitializer','he','Padding','Same')
             batchNormalizationLayer('Name','BN4')
             reluLayer('Name','relu4')       
               
             fullyConnectedLayer(2,'Name','fc1') 
             
             softmaxLayer('Name','sf')
             classificationLayer('Name','output')];
         
         lgraph = layerGraph(layers);        
           analyzeNetwork(lgraph)

 %Specifying the training options
     options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch');

% Training the network
netTransfer = trainNetwork(augimdsTrain,layers,options);

save('ConvNet18_Rimone2.mat')
toc;
 
%% PROGRAM for testing
 
clc; clear all; close all;

tic;

load('ConvNet18_Rimone2.mat')

% Classification validation
[YPred,scores] = classify(netTransfer,augimdsTest);

%Accuracy calculation
YValidation = FinalTest.Labels;
accuracy = mean(YPred == YValidation)

% Plot confusion matrix 
figure, plotconfusion(YValidation,YPred)

xlswrite('Rimone2_Results',scores,1);
writematrix(YPred,'Rimone2_Results.txt','Delimiter','tab')

toc;

%%  Grad cam visualization
Img1 = readimage(FinalTest,1);
Img1 = imresize(Img1,[64,64]);

figure(1),imshow(Img1)
label_Img1 = classify(netTransfer,Img1);

% Visulaization using Grad-CAM
scoreMap_Img1 = gradCAM(netTransfer,Img1,label_Img1);

figure(2),imshow(Img1)
hold on
imagesc(scoreMap_Img1,'AlphaData',0.5)
colormap jet
%%



 

