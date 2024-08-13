# Setting up SPRITE on Windows

Detailed instructions for setting up SPRITE on Windows are provided as follows.

Tested releases:

* **Windows 11**
* **Windows 10**

Important notes:

 * If you are a **code developer** and plan to make changes of the code and push the changes back to the repository on GitHub, we assume you have a GitHub account and you know how to set up SSH keys on your operating system and add your SSH keys to your GitHub account.
 * We have prepared training data that can be directly used for some data-driven models in SPRITE. It is not mandatory to have the existing training data for using the data-driven models in SPRITE. But generating high-quality training data which are large in file size would take a few hours. We encourage users to download our prepared training data  to get familiar with using the data-driven models in SPRITE. The training data are stored in a separate zip file called **SPRITE-Data**.

Follow the instructions below. We use **username** to denote your user name on your operating system. Replace **username** with your user name.

## Install the prerequisite software packages

Download and install [Git for Windows](https://gitforwindows.org/).

Download and install [Anaconda](https://www.anaconda.com/).

## Obtain the SPRITE repository from GitHub

Find and click open the *Git Bash* app from the *Search* bar.

<img src="figs/pic_icon_git_bash.png">

A terminal-like window will show up.

<img src="figs/pic_window_git_bash.png">

If you are an **end user**, copy and paste the following text into the Git Bash command line and click *Enter* key to download the SPRITE repository

	git clone https://github.com/idaholab/SPRITE.git

If you are a **code developer**, copy and paste the following text into the Git Bash command line and click *Enter* key to download the SPRITE repository

	git clone git@github.com:idaholab/SPRITE.git

Notes: If you are an end user, *Git Bash* is only used for downloading the repository. You will not need *Git Bash* for the following steps.

## Obtain SPRITE-Data from Box

Download the SPIRITE-Data.zip file from  https://inlbox.box.com/s/2d51ythhga9knwv0euuig215khbel3lg

Unzip SPRITE-Data.zip and make sure the SPRITE-Data folder is located alongside the SPRITE folder.

## Create and activate a custom Conda environment for SPRITE

Find and click open the *Anaconda Navigator* app from the *Search* bar.

<img src="figs/pic_icon_anaconda.png">

A graphical window will show up. Find *Powershell Prompt* and click *Launch*.

<img src="figs/pic_window_anaconda_navigator.png">

A *Powershell Prompt* window will show up.

<img src="figs/pic_window_powershell.png">

Go to the SPRITE folder: copy and paste the following text into the *Powershell Prompt* command line and click *Enter* key.

	cd ./SPRITE
	
To create a new environment named *sprite*, copy and paste the following text into the *Powershell Prompt* command line and click *Enter* key.

	conda create -n sprite python=3.9

To activate the *sprite* environment, copy and paste the following text into the *Powershell Prompt* command line and click *Enter* key.

	conda activate sprite

To verify if the *sprite* environment is ready, copy and paste the following text into the *Powershell Prompt* command line and click *Enter* key.

	conda env list

You should see the following command line output.

	# conda environments:
	#
	base                     C:\Users\username\anaconda3
    sprite                *  C:\Users\username\anaconda3\envs\sprite

The *sprite* environment is empty. To install the required packages in the *sprite* environment, copy and paste the following text into the *Powershell Prompt* command line and click *Enter* key.

	pip install -r requirements.txt

Now go back to [**How to use**](../) section.
