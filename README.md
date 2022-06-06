# Reinforcement Learning on CarRacing-V0 with Different Image-Based Observation Spaces 
(COMP3004 [Designing Intelligent Agents] Coursework)

### Project Description: 
This is a project to investigate the effects of using different types of image-based observation spaces as input when training reinforcement learning agents within the same environment.

![ezgif-5-d578298742](https://user-images.githubusercontent.com/49972425/172167564-a8fd1ddf-cbb7-41d9-a80e-b6d6e9818349.gif)

### To achieve this aim, a few objectives are met: 
1. A reinforcement learning algorithm is trained on different observation spaces, producing agents able to be used for evaluation. 
2. Agents evaluated to retrieve performance obtained at different timesteps of training for comparison. 
3. Learned behaviour of agents within the environment are inspected and recorded.

The PPO algorithm from was used for the experimentation.

### Observation Space Modifications experimented:
1. Colour Space Conversion

![CSC](https://user-images.githubusercontent.com/49972425/172168596-47dcd92b-3d56-49db-b999-caa0cea3f57f.PNG)

2. Frame Stacking

![Framestack](https://user-images.githubusercontent.com/49972425/172168593-ab9955db-40fb-47d0-aa23-633545ad3846.PNG)

3. Image Downsampling

![Downsampling](https://user-images.githubusercontent.com/49972425/172168599-ae0dd381-25a0-4601-840e-23828a4759dd.PNG)

### Results: 
Results obtained can be read on the ["Reinforcement Learning on CarRacing-V0 with Different Image-Based Observation Spaces"](https://github.com/khoryongteng/RL_Driving/blob/main/Reinforcement%20Learning%20on%20CarRacing-V0%20with%20Different%20Image-Based%20Observation%20Spaces.pdf) report.

### Directory Explanation:
1. PPO_Training files - Code used for training models on different Observation Spaces
2. PPO_Evaluation files - Code used for evaluating trained models, producing CSVs containing performance measures
3. Environment Folder - CarRacing-V0 environment and environment modification code
4. Evaluation Folder - CSVs containing performance measures of different models
5. Models Folder - Not added due to file size limitations

### To run the experiments yourselves:

Required Installations:
1. Swig
2. Python
3. Jupyter Notebook

Required Packages:
1. gym[box2d]==0.21.0
2. stable_baselines3[extra]==1.5.0
3. supersuit==3.3.3
4. tensorflow==2.8.0
5. pandas
6. matplotlib
7. numpy
8. seaborn
