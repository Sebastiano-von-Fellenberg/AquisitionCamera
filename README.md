# Aquistion Camera

The aquisition_cam.py is a python programm to reduce and analyze astronomic data.

Here is an example on how to call it in an ipython3 shell. This code only works with Python3.

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/call_aquisitioncam.png)

First, import all the classes and functions and then give the path where the data is stored.
The **opener()**-function opens all GRAVI.*fits files where S2 is in the science object and that are not sky images. It saves the aquisition image data in i and the headers in h. It prints out each individual path.

Now, the different classes can be called.

## ObservationNight
The ObservationNight reduces the images in the given path, determines the offset of star S65 in each image and creates a list with this offset in pixels, when called.
Then it alings all the images regarding to this offset and stacks them into a cube. There is one cube for the 5min exposure images and another one for the frames.

Here is an example of calling the class.

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/call_obs.png)


The functions that are executed when calling ObservationNight are:   
**get_reduction()**: reduces the images  
**get_reference_frame()**: determines offset  
**get_alignment()**: alings the images and saves them into cubes  

Another function is the **save()**-function. IT saves the raw data cube, the shifted images cube and the shifted frames cube as fits files. The files are saved into the path-directory unless another save directory is given.

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/obs_savedir.png)


## ObservationAnalysis 
The ObservationAnalysis class inherits from the ObservationNight class, so that all the function from ObservationNight can also be called in ObservationAnalysis. Call the class like this:

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/call_oba.png)

This class plots the lightcurve of Sgr A* in flux relative to the flux of a reference star when called. The reference star can be either S30, S65 or S10. This star is also used to calculate the position of Sgr A*. Change it with:

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/oba_which.png)

Here is an example for the lightcurve.

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/lightcurve_easter.png)


The functions that are executed when calling ObservationAnalysis are:   
**get_lightcurve_star(which)**: gets flux for star 'which'  
**get_lightcurve_Sgr(which)**: gets flux for Sgr A*. 


## Image
The Image class is a base class for astronomy images. It just saves the data images.


## AquisitionImage
The AquisitionImage class inherits from Image. It is used for reduction by subtracting the background and interpolating the dead pixels. It also gets the position of the science and fringe tracking fiber, as well as getting the time of each exposure based on dit and ndit.  
It can be called like this, by picking out one image and one header:   

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/call_aq.png)

Here is an example of the image after sky subtraction and dead pixel interpolation, of the image after sky subtraction and of the raw image.

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/aq_corrected.png)


## ScienceImage
The ScienceImage class inherits from AquisitionImage. It subtracts the background from the image and also determines the time for each image and each frame. It gets called like the AquisitionImage.  
Here is an example of the fully reduced image in regard to the one with the background, and also the background itself.

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/sc_imageback.png)


## GalacticCenterImage
The GalacticCenterImage class inherits from ScienceImage. It has an implemented starfinder which is able to detect sources in the image and creates a table with information like the x and y position in pixels and the flux. It also gives the name for several stars. It is called like the ScienceImage.  
Here is an example of how the star detection looks like.

![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/example_found_stars_names.png)
