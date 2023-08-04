	 ____    ____    ____    ___   _____   _____
	/ ___|  |  _ \  |  _ \  |_ _| |_   _| | ____|
	\___ \  | |_) | | |_) |  | |    | |   |  _|
	 ___) | |  __/  |  _ <   | |    | |   | |___
	|____/  |_|     |_| \_\ |___|   |_|   |_____|
	
	Smart Preprocessing & Robust Integration Emulator

# Setting up SPRITE

Detailed instructions for setting up SPRITE on the latest macOS are provided as follows.

## macOS

### Tested releases:

* **macOS Ventura** (*"macOS 13"*)
* **macOS Monterey** (*"macOS 12"*)

Follow the **Terminal** command line instructions below. In the example commend line instructions below, We use `username` to denote your user name on your operating system. Replace `username` with your user name.


### Installing the prerequisites

<!--
* **Important**: Open a Terminal window. Change the default **zsh** to **bash**:

		chsh -s /bin/bash
-->
Install Homebrew by following instructions on [https://brew.sh/](https://brew.sh/)
Use Homebrew to install wget:

	brew install wget

Use Homebrew to install git:

	brew install git
		
<!--
* Install **Conda**. Conda is an open-source, cross-platform, language-agnostic package manager and environment management system. To install Conda, follow the instruction on the Conda [website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#installation). For example, we use Miniconda for creating the needed Conda environment.

* To check if the system environment variables are set up properly for Conda, type:

		which conda
		
* You should see the following command line output:

		/Users/username/miniconda/bin/conda

* If not, copy and paste the following lines into the file `.bash_profile` located in `/Users/username/`. Remember to replace `username` with your user name.

		#... CONDA
		#!/bin/sh
		_CONDA_ROOT="/Users/username/miniconda"
		# Copyright (C) 2012 Anaconda, Inc
		# SPDX-License-Identifier: BSD-3-Clause
		\. "$_CONDA_ROOT/etc/profile.d/conda.sh" || return $?
		conda activate "$@"
		
* Save and quit the file `.bash_profile`. Close the Terminal window. Open a new Terminal window.
-->


### Downloading the SPRITE repository from GitHub

To obtain the SPRITE repository:

Navigate to a preferred directory, e.g., the `Downloads` folder:

	cd /Users/username/Downloads

Download by git clone.

	git clone https://github.com/yidongxiainl/SPRITE.git

<!--
* Download option 1b. Unzip the downloaded file. 

		wget https://github.com/yidongxiainl/SPRITE/archive/refs/heads/main.zip
	
* Download option 2 (recommended for developers):

		git clone git@github.com:yidongxiainl/SPRITE.git
-->

Note: We have prepared training data that can be directly used for some data-driven models in SPRITE. It is not mandatory to have the existing training data for using the data-driven models in SPRITE. But generating high-quality training data which are large in file size would take a few hours. We encourage users to download our prepared training data  to get familiar with using the data-driven models in SPRITE. The training data are stored in a separate GitHub repository called **SPRITE-SPRITEData**. To obtain the repository:

Navigate to the same directory where the SPRITE repository is located, e.g., the `Downloads` folder:

	cd /Users/username/Downloads

Download by git clone.

	git clone https://github.com/yidongxiainl/SPRITE-Data.git

<!--
* Download option 1b. Unzip the downloaded file. 

		wget https://github.com/yidongxiainl/SPRITE-Data/archive/refs/heads/main.zip
	
* Download option 2 (recommended for developers):

		git clone git@github.com:yidongxiainl/SPRITE-Data.git
-->
### Install Anaconda

Install **Anaconda**. Anaconda Navigator is a desktop graphical user interface (GUI) included in Anaconda® Distribution that allows you to launch applications and manage conda packages, environments, and channels without using command line interface (CLI) commands. Navigator can search for packages on Anaconda.org or in a local Anaconda Repository. To install Anaconda, download it from [the link](https://www.anaconda.com/download/), and follow the instruction on the Anaconda [website](https://docs.anaconda.com/free/anaconda/install/windows/).

### Activating the custom Conda environment for SPRITE
	
Create a new environment named sprite

	conda create -n sprite python=3.9
		
When it shows procceed([y]/n), press y

To activate the sprite environment, type

	conda activate sprite
		
You can also verify the activation of the SPRITE environment by typing

	conda env list
		
You should see the following command line output.

	# conda environments:
	#
	base                     /Users/username/anaconda
	sprite                *  /Users/username/anaconda/envs/SPRITE

Navigate to the SPRITE repository folder:

	cd Users/username/Downloads/SPRITE
		
To install the required packages of the sprite environment for SPRITE, type

	pip install -r requirements.txt

Now go back to [**How to use**](../) section.