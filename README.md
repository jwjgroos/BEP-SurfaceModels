# BEP-SurfaceModels
Comparison of two machine learning
approaches to direct seismic data imaging
using an U-Net convolutional neural network

Volledige BEP verslag kan aangevraagd worden

Code is used to create dummy surface models that can be used to train (and compare) a U-NET Convolutional Neural Network. 
Overview.png shows a flowchart of how the different codes are used to create the input data for the U-Net. 

# Quick Overview
Raw data + target data: Creates raw data shots and uses 10 iteration of migration to create the target data

Prefocusing: Creates prefocused shots with 1 iteration of migration as input images

1_ .py creates models 2_ .py creates raw data shots 3_ .py creates migrated reflecitivity images

4_ .py combines the images for the unet, saves them in 'U-Net'

Powershell can be used to automate the process seeds can be used to keep track of the models

# Overview  
1_Model_Making.py maakt de synthetische modellen. 

(removed) 2_FWMod.py maakt de ruwe data 

(removed) 3a_ is  voor het maken van de target images met 10 iteraties van migratie

(removed) 3b_  is voor het maken van de prefocused images deze gebruikt een aangepaste versie van de subroutines, anders werkte hij niet.

4a_ combineert alles zodat het in google drive gestopt kan worden voor mijn U-Net voor raw data approach.

4b_ zelfde maar dan voor prefocusing approach.

