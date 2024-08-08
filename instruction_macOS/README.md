	 ____    ____    ____    ___   _____   _____
	/ ___|  |  _ \  |  _ \  |_ _| |_   _| | ____|
	\___ \  | |_) | | |_) |  | |    | |   |  _|
	 ___) | |  __/  |  _ <   | |    | |   | |___
	|____/  |_|     |_| \_\ |___|   |_|   |_____|
	
	Smart Preprocessing & Robust Integration Emulator

# Setting up SPRITE on macOS

Tested releases:

* **macOS Ventura** (*"macOS 13"*)
* **macOS Monterey** (*"macOS 12"*)

Important notes:

 * If you are a **code developer** and plan to make changes of the code and push the changes back to the repository on GitHub, we assume you have a GitHub account and you know how to set up SSH keys on your operating system and add your SSH keys to your GitHub account.
 * We have prepared training data that can be directly used for some data-driven models in SPRITE. It is not mandatory to have the existing training data for using the data-driven models in SPRITE. But generating high-quality training data which are large in file size would take a few hours. We encourage users to download our prepared training data  to get familiar with using the data-driven models in SPRITE. The training data are stored in a separate GitHub repository called **SPRITE-Data**.

Follow the **Terminal** command line instructions below. We use **username** to denote your user name on your operating system. Replace **username** with your user name.

## Install the prerequisite software packages

Open a Terminal window. Change the default **zsh** to **bash**:

	chsh -s /bin/bash

Install [Homebrew](https://brew.sh/).

Use Homebrew to install git:

	brew install git

Download [Anaconda](https://www.anaconda.com/) for macOS by choosing the installer that matches your hardware, i.e., *Mac (Intel)* or *Mac (M1/M2)*. Double click the downloaded *pkg* file, e.g., *Anaconda3-2023.07-1-MacOSX-arm64.pkg*, to complete the installation.

## Obtain the SPRITE repository from GitHub

Go to a preferred directory, e.g., *Downloads*.

	cd ~/Downloads/

If you are an **end user**, use the https links to download the SPRITE repository

	git clone https://github.com/yidongxiainl/SPRITE.git

If you are a **code developer**, use the SSH links to download the SPRITE repository

	git clone git@github.com:yidongxiainl/SPRITE.git

## Obtain SPRITE-Data from Box

Download the SPIRITE-Data.zip file from  https://inlbox.box.com/s/2d51ythhga9knwv0euuig215khbel3lg

Unzip SPRITE-Data.zip and make sure the SPRITE-Data folder is located alongside the SPRITE folder.

## Create and activate a custom Conda environment for SPRITE

Go to the SPRITE folder, e.g.,

	cd ~/Downloads/SPRITE
	
Create a new environment named *sprite*, type

	conda create -n sprite python=3.9

To activate the *sprite* environment, type

	conda activate sprite

You can verify if the *sprite* environment is ready by typing

	conda env list

You should see the following command line output.

	# conda environments:
	#
	base                     /Users/username/anaconda3
	sprite                *  /Users/username/anaconda3/envs/SPRITE

The *sprite* environment is empty. To install the required packages in the *sprite* environment, type

	pip install -r requirements.txt

Now go back to [**How to use**](../) section.
