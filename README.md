# MRBDTA
A  deep learning model for predicting drug-target binding affinity
1 System requirements:

	Hardware requirements: 
		Model.py requires a computer with enough RAM to support the in-memory operations.
		Operating system：windows 10

	Code dependencies:
		python '3.7.4' (conda install python==3.7.4)
		pytorch-GPU '1.10.1' (conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch)
		numpy '1.16.5' (conda install numpy==1.16.5)

2 Installation guide:

	First, install CUDA 10.2 and CUDNN 8.2.0.
	Second, install Anaconda3. Please refer to https://www.anaconda.com/distribution/ to install Anaconda3.
	Third, install PyCharm. Please refer to https://www.jetbrains.com/pycharm/download/#section=windows.
	Fourth, open Anaconda Prompt to create a virtual environment by the following command:
		conda env create -n env_name -f environment.yml

	Note: the environment.yml file should be downloaded and put into the default path of Anaconda Prompt.

3 Instructions to run Model.py:

	Based on kiba dataset:
		First, put folder data_kiba, DataHelper.py, emetrics.py and Model.py into the same folder.
		Second, use PyCharm to open Model.py and set the python interpreter of PyCharm.
		Third, modify codes in Model.py to set the path for loading data and the path for saving the trained model. The details are as follows:
			line 249 in Model.py
			line 268 in Model.py
		Fourth, open Anaconda Prompt and enter the following command:
			activate env_name
		Fifth, run Model.py in PyCharm.

		Expected output：
			The kiba scores between drugs and targets in test set of kiba dataset would be output as a csv file.

		Expected run time on a "normal" desktop computer:
			The run time in our coumputer (CPU:Xeon 3106, GPU NVIDIA Geforce RTX 2080 Ti, ARM 64G) is about 24 hours.
	
	Based on davis dataset:
		First, put folder data_davis, DataHelper.py, emetrics.py and Model.py into the same folder.
		Second, use PyCharm to open Model.py and set the python interpreter of PyCharm.
		Third, modify codes in Model.py to set parameters for the davis dataset. The details are as follows:
			line 241 to line 248 in Model.py: 'max length for drugs, max length for proteins, trian set, test set'.
			line 253 in Model.py: 'drug, target, affinity = DH.LoadData(fpath_kiba, logspance_trans=False)' -> 'drug, target, affinity = DH.LoadData(fpath_davis, logspance_trans=True)'.
			line 265 in Model.py: 'EPOCHS, batch_size, accumulation_steps = 600, 32, 32' -> 'EPOCHS, batch_size, accumulation_steps = 300, 32, 8'.
		Fourth, modify codes in Model.py to set the path for loading data and the path for saving the trained model. The details are as follows:
			line 244 in Model.py
			line 268 in Model.py
		Fifth, open Anaconda Prompt and enter the following command:
			activate env_name
		Sixth, run Model.py in PyCharm.

		Expected output：
			The Kd in nM between drugs and targets in test set of davis dataset would be output as a csv file.

		Expected run time on a "normal" desktop computer:
			The run time in our coumputer (CPU:Xeon 3106, GPU NVIDIA Geforce RTX 2080 Ti, ARM 64G) is about 6 hours.
