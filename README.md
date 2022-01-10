# SepPCNET
Environ. Sci. Technol. 2021, 55, 9958−9967

The structural description in the reported DL-QSAR models is still restricted to the twodimensional level. Inspired by point clouds, a type of geometric data structure, a novel three-dimensional (3D) molecular surface point cloud with electrostatic potential (SepPC) was proposed to describe chemical structures. Each surface point of a chemical is assigned its 3D coordinate and molecular electrostatic potential. A novel DL architecture SepPCNET was then introduced to directly consume unordered SepPC data for toxicity classification. The SepPCNET model was trained on 1317 chemicals tested in a battery of 18 estrogen receptor-related assays of the ToxCast program. The obtained model recognized the active and inactive chemicals at accuracies of 82.8 and 88.9%, respectively, with a total accuracy of 88.3% on the internal test set and 92.5% on the external test set, which outperformed other up-to-date machine learning models and succeeded in recognizing the difference in the activity of isomers. Additional insights into the toxicity mechanism were also gained by visualizing critical points and extracting data-driven point features of active chemicals.


Installation:
Install PyTorch. The code was used with python3.7,PyTorch. 

Usage: 
To train a model to classify environmental estrogens by 3D surface electrostatic potential point cloud:
python SepPCNET.py
Loss, AUC-ROC on training set and validation set will be saved as txt files and network parameters after each training epoch will be saved.

Hyperparameter:
lr weight_decay batch_size of the best model in this work are used in SepPCNET.py indefault.
The training process would be stopped if no increase in prediction AUC-ROC on the validation set was observed for 15 consecutive epochs.

Example:
50-28-2sparse.txt and 57-91-0sparse.txt are examples of point cloud input files. Each row represents a point in point cloud of 4096 points.
