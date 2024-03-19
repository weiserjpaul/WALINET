import sys
from LipIDCNN.GenerationFID import Gene_FID_withLipids
sys.path.insert(0, '/autofs/space/somes_002/users/pweiser/lipidSuppressionCNN/make_CNN_StackData_rrrt/LipIDCNN/GenerationFID')

CNNTrainingDataPath = 'data/vol22/CNNLipTrainingData_vol22.h5'
NameStackLipDataPath = 'data/vol22/vol22_LipStack_100MB.h5'
NameLipOpPath = 'data/vol22/vol22_LipProj.h5'
NbTrainEx = 5000
MaxLipScaling = 500
FieldStrength = 7

if FieldStrength==3:
    MaxFreq_Shift=50
    MaxPeak_Width=30
    MinBLWidth =200
elif FieldStrength==7:
    MaxFreq_Shift=100
    MaxPeak_Width=60
    MinBLWidth =400



Gene_FID_withLipids.main(['-o', CNNTrainingDataPath, 
                            '-l', NameStackLipDataPath, 
                            '-lop', NameLipOpPath, 
                            '--ntrain', str(NbTrainEx),
                            '--maxLipSc', str(MaxLipScaling), 
                            '--maxFShft', str(MaxFreq_Shift), 
                            '--maxPkW', str(MaxPeak_Width), 
                            '--wBL', str(MinBLWidth),
                            '--maxWatSc', str(0),
                            '-vf'])