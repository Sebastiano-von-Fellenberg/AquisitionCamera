# Aquistion Camera




## ObservationNight
Call it with: o = ObservationNight(path).
When called, the class will reduce the images in the given path, determine the offset of star S65 in each image and create a list with this offset in pixels.
Then it alings all the images regarding ton this offset and stacks them into a cube. There is one cube for the 5min exposure images and another one for the frames.

Functions:
**get_reduction()**: reduces the images
**get_reference_frame()**: determines offset
**get_alignment()**: alings the images and saves them into cubes
**save()**: saves the raw cube, the shifted images cube and the shifted frames cube as fits file. Does not automatically get called when calling the class. Will save the cubes in path if no other save directory (savedir) is given.

## ObservationAnalysis 
Call it with: o = ObservationAnalysis(path)
Determines the flux for a reference star (kwarg: which)


![alt text](https://github.com/Sebastiano-von-Fellenberg/AquisitionCamera/blob/giulia_branch/example_found_stars_names.png)
