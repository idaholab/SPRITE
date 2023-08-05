	 ____    ____    ____    ___   _____   _____
	/ ___|  |  _ \  |  _ \  |_ _| |_   _| | ____|
	\___ \  | |_) | | |_) |  | |    | |   |  _|
	 ___) | |  __/  |  _ <   | |    | |   | |___
	|____/  |_|     |_| \_\ |___|   |_|   |_____|
	
	Smart Preprocessing & Robust Integration Emulator

&copy; 2023 Idaho National Laboratory

**SPRITE (Smart Preprocessing & Robust Integration Emulator)** is an open-source suite of analytical and data-driven models for predicting the performance of renewable carbon feedstock preprocessing units and system integration. The models being developed in this suite include **Population Balance Model (PBM)**, **Enhanced Deep Neural Operator (DNO+)**, and **Physics-Informed DNO+ (PIDNO+)**.

# Installing SPRITE

Detailed instructions for installing the prerequisite software packages and SPRITE on the different operating systems, e.g., Windows, macOS and Linux Ubuntu LTS releases, are provided. Click the link(s) below to learn about the installation process based on your operating systems.

* [Windows](instruction_Windows/)
* [Linux Ubuntu](instruction_Linux_Ubuntu/)
* [macOS](instruction_macOS/)

# Data description
The source experimental data was obtained from two mechanical size reduction mills: a Wiley knife mill at the bench scale and a JRS knife mill at the pilot scale. The data contains the cumulative output particle size distributions (PSDs) under different conditions of processing parameters. Find the description [here](PBM/Dataset).


# Use SPRTIE

## Running SPRITE
1. For Windows users, open PowerShell app in Anaconda Navigator. For macOS/Linux users, open the terminal. 
2. Go to the project folder where the file - '/SPRITE/main.py' is located.

	Windows
		
		cd c:/Users/username/Downloads/SPRITE
			
	macOS

		cd /Users/username/Downloads/SPRITE
			
	Run SPRITE
		
		./RUN
			
	Three models can be used to predict the milled biomass particle size distribution (PSD) with the given feed biomass PSD under certain milling condition.

	- PBM: A probabilistic based model, by data fitting and regression to find the input output relationship. It has mode 1 to show the data fitting performance, and has mode 2 for prediction. In prediction mode, you can use the model to predict exist data from the experiment, or make a new data for it to predict.
	- DNO+: A data driven based model, trained by 300 datasets, 100 test datasets. You can try the training process, or use the pre-trained model to make prediction. In prediction mode, you can use the model to predict exist data from the experiment, or make a new data for it to predict.
	- PIDNO+: A phyisics-informed and data driven based model, trained by 25 datasets, 5 datasets for testing. You can try the training process, or use the pre-trained model to make prediction.

## Test model on new data
### Create new data
1. First prepare the feed biomass particle size distribution (PSD) as the input data and the milled product biomass PSD as the output data. The data should include the sieve sizes and the cumulative PSD, in the form of .csv file below:

	* \SPRITE\PBM\Dataset\extra_input.csv
	* \SPRITE\PBM\Dataset\extra_output.csv

	You can directly put your data into these two files, under the column name: 'FSieves' & 'Feedmass' in the input, 'ExpSieves' & 'Expmass' in the output.

2. Run SPRITE, and select Method 1 - PBM
3. When it shows: 'Enter the No. data you want choose (choose between 1~10) or enter 0 for your own data.', press 0.
4. Then follow the instruction in the code.
	* Enter the milling frequency in Hz: 
	* Enter the moisture content in %: 
5. Select the optimization method.
6. Finish fitting process and get the results.


# Troubleshoot
1. "PermissionError: [Errno 1] Operation not permitted"
	- Conditions: On Mac OS, try to use the model load with DNO+ or PIDNO+ methods under terminal or other python interface.
	- Solver: Open Mac system settings -- Privacy & Security -- Full Disk Access. 
		- If your app is in the list: give the disk access to it directly.
		- If your app is not in the list: click on '+' button below, find your app, then give the disk access.

2. "No module named 'XXX'"
	- Conditions: Using console, and try to use different models under the same console.
	- Solver: Restart console. 