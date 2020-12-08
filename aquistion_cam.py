import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from glob import glob
from datetime import datetime
import gc
import pandas as pd

from astropy.stats import SigmaClip
from astropy.time import Time as AstropyTime
from astropy.table import Column, Table
from astropy.io import ascii
from photutils import Background2D, MedianBackground

from scipy.signal import correlate2d, convolve2d
from scipy.ndimage import shift
from photutils.centroids import centroid_2dg

from matplotlib.colors import LogNorm


PIXEL_SCALE                      = 17.89 #mas/px Anugu+2018 




def opener(path):
    """
    helper files that opens all files with GRAVI.*fits that are not skys or S2 is not in the science object
    """

    headers                     = []
    aquistion_images            = []
    names                       = []
    
    #print(sorted(glob(path + "GRAVI.*fits"))[:9])
    for fi in sorted(glob(path + "GRAVI.*_dualscip2vmred.fits")):
        #print(fi)
        if "aqu" not in fi and fi != path + 'GRAVI.2019-04-22T09:23:13.816.fits' and fi != path + 'GRAVI.2019-04-22T09:29:13.832.fits':
            h0                      = fits.open(fi)[0].header 
            if "S2" in h0["ESO INS SOBJ NAME"] and "SKY" not in h0["HIERARCH ESO OBS NAME"]:
                i0                      = fits.open(fi)[17].data 
                print(h0["ARCFILE"])
                headers.append(h0)
                aquistion_images.append(i0)
    return aquistion_images, headers


class ObservationNight(list):
    def __init__(self, path, gc=True, savedir=None, test=False, verbose=False):
        """
        Reduces all images in the path
        args:
        path  str, path to the dir where the GRAVITY fits are located
        
        kwargs:
        gc bool, if True assuming GalacticCenterImages
        savedir, None or str. if savedir is None savedir is set to path, else to the value of savedir
        
        functions:
        self.get_reduction()
        reduces the images by creating either ScienceImage image_objects or if gc==True GalacticCenterImage
        
        sets attributes
            self.image_objects, list list of image objects reduced 
            
        self.save():
            calls the save_fits function of ScienceImage to the path stored in self.savedir
            
        self.aligncube(x0, y0)
        args
        x0, y0 pixel position of the alignment star
        
        aligns the upe according to the position x0, y0
        returns the necessary shifts to align the cube
        
        self.computeshift(shift_list)
        computes median and std of shift_list to average out the position shifts (currently not implemented)
        returns the median and std
        
        self.shiftcube(x0=28, y0=13, search_box=10):
        shifts the cubes accroding to x0, y0. Default 28, 13 which should be s35 if the observations are normal
        search_box is the search range in which S35 is searched (or the star at x0, y0)
        returns the shifted cube
        
        self.get_reference_frame()
        determines position of S65 for each image 
        creates list with displacement according to self.x0, self.y0 (estimate of S65 position)
        sets attributes:
            self.position_S65, self.mask_S65, self.mask_S30, self.shifted_list
            
        self.get_alignment()
        shifts images and frames into cube according to self.shifted_list
        sets time in [:,0,0] and mask in [:,0,1]
        sets attributes:
            self.shifted_images, self.shifted_frames, self.time_images, self.time_frames
        """
        self.path               = path
        self.gc                 = gc
        self.test               = test
        self.verbose            = verbose
        
        self.image_objects      = None
        
        if savedir is None:
            self.savedir        = path
        else:
            self.savedir        = savedir
        
        self.position_S65 = None
        self.shift_list = None
        self.mask_S65 = None
        self.mask_S30 = None
        
        self.x0 = 10
        self.y0 = 20
        
        self.aquistion_images, self.headers = opener(self.path)
        self.get_reduction()
        self.get_reference_frame()
        self.get_alignment()

    def get_reduction(self):
        self.image_objects      = []
        if self.gc:
            for i, h in zip(self.aquistion_images, self.headers):
                o               = GalacticCenterImage(i, h)
                o.get_PSF()
                self.image_objects.append(o)
        else:
            for i, h in zip(self.aquistion_images, self.headers):
                self.image_objects.append(ScienceImage(i, h))
        del(o)
        gc.collect()
    
    def save(self, overwrite=False, save_dir=None):
        print("create an a fits extentation and an asci table.")
        if save_dir is None:
            save_dir            = self.save_dir
        for io in self.image_objects:
            io.save_fits(save_dir, overwrite=overwrite)

        nightcube               = []
        for io in self.image_objects:
            nightcube.append(io.image)
        self.nightcube          = np.array(nightcube)
        
        fits.writeto(save_dir + "aquistion_raw_cube.fits", np.array(nightcube), overwrite=overwrite)
        fits.writeto(save_dir + "aquistion_shifted_cube.fits", np.array(self.shifted_images), overwrite=overwrite)
        fits.writeto(save_dir + "aquistion_shifted_frames_cube.fits", np.array(self.shifted_frames), overwrite=overwrite)
        
        flux_table              = Table(self.flux_aperture)
        ascii.write(flux_table, save_dir + "lightcurve_aperture.csv")
        
    def aligncube(self, x0, y0, search_box=5):
        positions = []
        shifted_cube =[]
        for image in self.nightcube:
            positions.append(centroid_2dg(image[x0-search_box:x0+search_box, y0-search_box:y0+search_box]))
        positions = np.array(positions)
        return positions[0] - positions
        
    def computeshift(self, shift_list):
        #positions[0][1] - p[1], positions[0][0] - p[0] ## this once worked!!
        return np.median(shift_list, axis=0), np.std(shift_list, axis=0)

    def shiftcube(self,x0=28, y0=13, search_box=10):
        shift_list              = self.aligncube(x0, y0, search_box=search_box)
        shifted_cube = []
        for image, s in zip(self.nightcube, shift_list):
            shifted_cube.append(shift(image, (s[1], s[0])))
            
        return np.array(shifted_cube)
    
    def get_reference_frame(self):
        """
        calculates displacement of S65 in images
        """
        self.position_S65 = []
        self.shifted_list = []
        self.mask_S65 = []
        self.mask_S30 = []
        for obj in self.image_objects:
            #print(obj.S65)
            #obj.get_stars()
            if obj.S65 == None:
                self.position_S65.append((0., 0.))
                self.mask_S65.append(False)
            else:
                self.position_S65.append((obj.S65["xcentroid"], obj.S65["ycentroid"]))
                self.mask_S65.append(True)
            
            self.shifted_list.append((self.x0 - self.position_S65[-1][0], self.y0 - self.position_S65[-1][1]))
            
            if self.test:
                plt.imshow(obj.image, origin='lower', norm=LogNorm(vmax=50))
                plt.plot(self.position_S65[-1][0], self.position_S65[-1][1], 'o')
                plt.show()

            if obj.S30 == None:
                self.mask_S30.append(False)
            else:
                self.mask_S30.append(True)
        
    def get_alignment(self):
        """
        algins all images, so that sources are on same spot
        """
        shifted_images_mean = []
        shifted_images_frames = []
        self.time_images = []
        self.time_frames = []
        
        for index, image_object in enumerate(self.image_objects):
            shifted_image = shift(image_object.image, (self.shifted_list[index][1], self.shifted_list[index][0]))
            t = AstropyTime(str(image_object.date + np.timedelta64(int(np.round(image_object.ndit/2*image_object.dit*1000)), "ms"))).decimalyear
            shifted_image[0,0] = np.array((t))
            shifted_image[0,1] = np.array((self.mask_S65[index]))
            shifted_images_mean.append(shifted_image)
            self.time_images.append(t)

            for f, f_t in zip(image_object.frames, image_object.frame_times):
                shifted_frame = shift(f, (self.shifted_list[index][1], self.shifted_list[index][0]))
                time = AstropyTime(str(f_t)).decimalyear
                shifted_frame[0,0] = np.array((time))
                shifted_frame[0,1] = np.array((self.mask_S65[index]))
                self.time_frames.append(time)
                shifted_images_frames.append(shifted_frame)

            if self.test:                
                plt.imshow(shifted_image, origin="lower", norm=LogNorm(vmax=50, vmin=10))
                plt.plot(self.position_S65[index][0], self.position_S65[index][1], "o")
                plt.plot(self.x0, self.y0, "o")
                plt.show()
        
        self.shifted_images = np.array(shifted_images_mean)
        self.shifted_frames = np.array(shifted_images_frames)

        if self.test:
            plt.show()
            plt.close()
        
class ObservationAnalysis(ObservationNight):
    def __init__(self, path, gc=True, savedir=None, test=False, verbose=False):
        """
        Generates lightcurve for one observation night
        args:
        path

        kwargs:
        test: bool if True implented autotests are preformed
        verbose: bool if True code becomes talkative
        
        functions:        
        self.get_lightcurve_star()
        kwarg: which (S30, S65, S10)
        gets flux values for reference star which by aperture on its position from self.sources 
        sets attributes:
            self.flux_aperture, self.flux_starfinder
            
        self.get_lightcurve_Sgr()
        gets flux from SgrA* by aperture based on position of which and distance calculated from GalacticCenterImage.compute_distance_star()
        """
        super().__init__(path, gc=gc, savedir=savedir, test=test, verbose=verbose)
        self.flux_aperture = dict()
        self.flux_starfinder = dict()
        self.get_lightcurve_star()
        self.get_lightcurve_star(which="S65")
        self.get_lightcurve_Sgr()
    
    def get_lightcurve_star(self, which='S30'):
        """
        gets flux values for which
        """
        flux_aperture = []
        flux_starfinder = []
        times           = []
        mask = np.zeros((100,100))
        mask_bkg = np.zeros((100,100))
        
        for obj in self.image_objects:
            #obj.get_stars()
            star = getattr(obj, which)
            
            mask = np.zeros((100,100))
            mask_bkg = np.zeros((100,100))
            
            if star == None:
                for f in obj.frames:
                    flux_aperture.append(np.nan)
                    flux_starfinder.append(np.nan)
                pass
            
            else:
                x = int(np.round(star['ycentroid']))
                y = int(np.round(star['xcentroid']))

                mask[x-3:x+4,y] = 1
                mask[x,y-3:y+4] = 1 
                mask[x-2:x+3,y-2:y+3] = 1
                
                mask_bkg[x-7:x+8,y] = 1
                mask_bkg[x,y-7:y+8] = 1
                mask_bkg[x-6:x+7,y-4:y+5] = 1
                mask_bkg[x-4:x+5,y-6:y+7] = 1
                mask_bkg[x-5:x+6,y-5:y+6] = 1
                mask_bkg[x-5:x+6,y] = 0
                mask_bkg[x,y-5:y+6] = 0
                mask_bkg[x-4:x+5, y-4:y+5] = 0
                
                for f, t in zip(obj.frames, obj.timeframes):
                    #obj.get_stars()
                    flux_starfinder.append(star['flux'])
                    star_object = np.multiply(mask, f)
                    background = np.multiply(mask_bkg, f) * (mask.sum()/mask_bkg.sum())
                
                    flux_aperture.append(star_object.sum()-background.sum())
                    times.append(t)
                    if self.test:
                        plt.figure()
                        plt.imshow(obj.image, origin='lower', norm=LogNorm(vmax=100))
                        plt.imshow(star_object, origin="lower", alpha=0.1, norm=LogNorm(vmax=100))
                        plt.figure()
                        plt.imshow(obj.image, origin='lower', norm=LogNorm(vmax=100))
                        plt.imshow(np.multiply(mask_bkg, obj.image), origin="lower", alpha=0.1, norm=LogNorm(vmax=100))
                        plt.show()

        self.flux_starfinder[which + "_flux"] = np.array(flux_starfinder)
        self.flux_starfinder["time"]      = np.array(times)
        self.flux_aperture[which +"_flux"] = np.array(flux_aperture)
        self.flux_aperture["time"]      = np.array(times)
    def get_lightcurve_Sgr(self, which='S30'):
        """
        gets flux values for SgrA*
        """
        flux_Sgr = []
        mask_bkg = np.zeros((100,100))
        

        

        def mask_function(x,y):
            mask = np.zeros((100,100))
            mask[x-2:x+3,y] = 1
            mask[x,y-2:y+3] = 1
            mask[x-1:x+2,y-1:y+2] = 1
            return mask 
        
        for obj in self.image_objects:
            #obj.get_stars()
            star = getattr(obj, which)
            distance_star = obj.compute_distance_star(which=which)
            if star == None:
                for f in obj.frames:
                    flux_Sgr.append(np.nan)
                pass
            
            else:
                x_star = star['xcentroid']
                y_star = star['ycentroid']
                
                x_Sgr = x_star - distance_star[0]
                y_Sgr = y_star - distance_star[1]
  
                x = int(np.round(y_Sgr))
                y = int(np.round(x_Sgr))
                
                mask = mask_function(x, y)
                               
                mask_bkg[x-10:x+10,y-30:y-20] = 1
                
                background = np.multiply(mask_bkg, obj.image) * (mask.sum()/mask_bkg.sum())
                star_mask = np.multiply(mask, obj.image)
                
                for f in obj.frames:
                    #obj.get_stars()
                    star_object = np.multiply(mask, f)
                    background = np.multiply(mask_bkg, f) * (mask.sum()/mask_bkg.sum())
                    flux_Sgr.append(star_object.sum()-background.sum())
                
                if self.test:   
                    plt.figure()
                    plt.plot(x_star, y_star,"o", zorder=20)
                    plt.plot(x_Sgr, y_Sgr, "o", zorder=20)
                    plt.imshow(obj.image, origin='lower', norm=LogNorm(vmax=100))
                    plt.imshow(star_mask, origin='lower', norm=LogNorm(vmax=100))
                    plt.imshow(np.multiply(mask_bkg, obj.image), origin="lower", alpha=0.5, norm=LogNorm(vmax=100))
                    plt.show()
                    
                if self.test:
                    print("Position "+ which+": ", x_star, y_star)
                    print("Position SgrA*: ", x_Sgr, y_Sgr)
             
        self.flux_aperture["SgrA*_flux"] = np.array(flux_Sgr)
        
        plt.plot(self.time_frames, self.flux_aperture["SgrA*_flux"]/self.flux_aperture[which +"_flux"], '.')
        plt.ylabel('Flux relative to '+which)
        plt.xlabel('Time')
        plt.show()

class Image():
    def __init__(self, image, test=False, verbose=False):
        """
        Base class for astronomy images
        
        args:
        image: a np.ndarray array. Test for up to 3D
        
        kwargs:
        test: bool if True implented autotests are preformed
        verbose: bool if True code becomes talkative
        """
        self.image              = image
        if type(test) != bool or type(verbose) != bool: raise ValueError("Type of test and verbose needs to be boolean but is : ", type(test), type(verbose))
                                                                         
        self.test, self.verbose = test, verbose
        
class AquisitionImage(Image):
    def __init__(self, image, header, test=False, verbose=False, correct_dead_pixels=True):   
        """
        Base class for Aquistion Camera Images
        args:
        image: a np.ndarray array. Test for up to 3D
        header: a fits header image_object

        kwargs:
        correct_dead_pixels: bool, if True dead pixels stored in the deadpixel mask are interpolated default: True
        test: bool if True implented autotests are preformed
        verbose: bool if True code becomes talkative
        
        functions:
        self.get_image():
        function that cuts out the AC image
        sets attributes:
            self.image
        
        self.get_fiber():
        reads fiber information from header
        sets attributes:
            self.ft_fiber fringetracker fiber
            self.sc_fiber science fiber
            self.ref_aqu  infromation on where the AC image starts on the aquistion detector
        
        self.get_sc_pos():
        gets the postion of the science fiber in all four telescopes
        sets attributes:
            self.pos_sc     integer postion of the science fiber
            self.pos_sc_float float postion of the science fiber
            
        self.get_ft_pos():
        gets the postion of the ft fiber in all four telescopes
        sets attributes:
            self.pos_sc     integer postion of the ft fiber
            self.pos_sc_float float postion of the ft fiber
            
        self.get_time():
        gets the time of each exposure based on dit and ndit
        sets attributes:
            self.timestamps datetime timestamps based on cumsum(ndit_j*dit_j)
            self.timestamps_str same as strings
        
        self.load_bad_pixelmask():
        gets the bad pixel mask stored in the module
        
        self.load_dark():
        loads dark image
        sets attributes:
            self.dark
        
        sets attributes:
            self.mask the mask (np.ndarray)
        
        self.get_interpolation():
        interpolates the dead pixels and subtracts dark
        sets:
            self.image_uncorrected the raw image
            
        updates:
            self.image now with the dead pixel correction
        """
        super().__init__(image, test=test, verbose=verbose)
        self.raw_image          = self.image.copy()
        self.header             = header
        
        self.date               = np.datetime64(datetime.strptime(self.header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S"))
        self.mjd                = self.header["MJD-OBS"]
        self.exptime            = self.header["ESO DET2 SEQ1 DIT"]*self.header["ESO DET2 NDIT"]
        self.name               = self.header["ARCFILE"]
        self.ndit               = self.header["ESO DET1 NDIT"]
        self.dit                = self.header["ESO DET1 SEQ1 DIT"]
        

        
    
        
        self.get_image() 
        self.get_fiber()
        self.get_sc_pos()
        self.get_ft_pos()
        self.get_time()
        self.load_bad_pixelmask()
        self.load_dark()
        self.image_uncorrected = None
        if correct_dead_pixels:
            if self.verbose: print("Deadpixel correction, this takes some time... and comsumes fuck all memory")
            self.get_interpolation()
        
    def get_fiber(self):
        self.ft_fiber           = np.array([[self.header["HIERARCH ESO ACQ FIBER FT1X"], self.header["HIERARCH ESO ACQ FIBER FT1Y"]],
                                            [self.header["HIERARCH ESO ACQ FIBER FT2X"], self.header["HIERARCH ESO ACQ FIBER FT2Y"]],
                                            [self.header["HIERARCH ESO ACQ FIBER FT3X"], self.header["HIERARCH ESO ACQ FIBER FT3Y"]],
                                            [self.header["HIERARCH ESO ACQ FIBER FT4X"], self.header["HIERARCH ESO ACQ FIBER FT4Y"]]])
        
        self.sc_fiber           = np.array([[self.header["HIERARCH ESO ACQ FIBER SC1X"], self.header["HIERARCH ESO ACQ FIBER SC1Y"]],
                                            [self.header["HIERARCH ESO ACQ FIBER SC2X"], self.header["HIERARCH ESO ACQ FIBER SC2Y"]],
                                            [self.header["HIERARCH ESO ACQ FIBER SC3X"], self.header["HIERARCH ESO ACQ FIBER SC3Y"]],
                                            [self.header["HIERARCH ESO ACQ FIBER SC4X"], self.header["HIERARCH ESO ACQ FIBER SC4Y"]]])
        
        self.ref_aqu            = np.array([[self.header["ESO DET1 FRAM1 STRX"], self.header["ESO DET1 FRAM1 STRY"]],
                                            [self.header["ESO DET1 FRAM2 STRX"], self.header["ESO DET1 FRAM2 STRY"]],
                                            [self.header["ESO DET1 FRAM3 STRX"], self.header["ESO DET1 FRAM3 STRY"]],
                                            [self.header["ESO DET1 FRAM4 STRX"], self.header["ESO DET1 FRAM4 STRY"]]])
        
    def get_sc_pos(self):
        self.pos_sc_float       = np.array([[self.sc_fiber[0][1] - self.ref_aqu[0][1], self.sc_fiber[0][0] - self.ref_aqu[0][0]],
                                            [self.sc_fiber[1][1] - self.ref_aqu[1][1], self.sc_fiber[1][0] - self.ref_aqu[1][0] + 250],
                                            [self.sc_fiber[2][1] - self.ref_aqu[2][1], self.sc_fiber[2][0] - self.ref_aqu[2][0] + 500],
                                            [self.sc_fiber[2][1] - self.ref_aqu[2][1], self.sc_fiber[2][0] - self.ref_aqu[2][0] + 750]])
        self.pos_sc             = np.round(self.pos_sc_float).astype(int)
        
        self.pos_sc_float       = np.array([[self.sc_fiber[0][1] - self.ref_aqu[0][1], self.sc_fiber[0][0] - self.ref_aqu[0][0]],
                                            [self.sc_fiber[1][1] - self.ref_aqu[1][1], self.sc_fiber[1][0] - self.ref_aqu[1][0] + 250],
                                            [self.sc_fiber[2][1] - self.ref_aqu[2][1], self.sc_fiber[2][0] - self.ref_aqu[2][0] + 500],
                                            [self.sc_fiber[2][1] - self.ref_aqu[2][1], self.sc_fiber[2][0] - self.ref_aqu[2][0] + 750]])
    
    def get_ft_pos(self):
        self.pos_ft_float       = np.array([[self.ft_fiber[0][1] - self.ref_aqu[0][1], self.ft_fiber[0][0] - self.ref_aqu[0][0]],
                                            [self.ft_fiber[1][1] - self.ref_aqu[1][1], self.ft_fiber[1][0] - self.ref_aqu[1][0] + 250],
                                            [self.ft_fiber[2][1] - self.ref_aqu[2][1], self.ft_fiber[2][0] - self.ref_aqu[2][0] + 500],
                                            [self.ft_fiber[2][1] - self.ref_aqu[2][1], self.ft_fiber[2][0] - self.ref_aqu[2][0] + 750]])
        self.pos_ft             = np.round(self.pos_ft_float).astype(int)
        
    def get_image(self):
        self.image              = self.raw_image[:, 0:249, :] 
        
    def get_bad_pixelmask(self, ):
        def rms(image):
            im                  = np.array(image, dtype=float)
            im2                 = im**2
            ones                = np.ones(im.shape)
            kernel              = np.ones((3,3))
            kernel[0,2]         = 0
            kernel[0,2]         = 0
            kernel[2,0]         = 0
            kernel[2,2]         = 0
            kernel[1,2]         = 0.5
            kernel[0,1]         = 0.5
            kernel[2,1]         = 0.5
            kernel[1,2]         = 0.5
            s                   = np.array([convolve2d(i, kernel, mode="same") for i in im])
            s2                  = np.array([convolve2d(i2, kernel, mode="same") for i2 in im2])
            ns                  = np.array([convolve2d(o, kernel, mode="same") for o in ones])
            return np.sqrt((s2 - s**2 / ns) / ns)
        
        
        raise Exception("buggy implented")
        m0                      = rms(self.image)
        if self.test:
            fig, axes           = plt.subplots(1,2)
            axes[0].imshow(np.nanmedian(m0, axis=0), origin="lower", norm=LogNorm())
            axes[1].imshow(np.nanmedian(self.image, axis=0), origin="lower", norm=LogNorm())
            plt.show()
            
        self.mask               = np.ones_like(self.image, dtype=bool)
        #self.mask[m0 > 1e3]    = False
        self.mask[self.image <= 0.] = False
        self.mask[self.image >= 1e4] = False
        #plt.figure()
        #plt.imshow(np.median(self.mask,axis=0), origin="lower")
        #plt.show()
       
    def load_bad_pixelmask(self):
        data                    = fits.getdata("gvacq_DeadPixelMap.fits")
        mask                    = np.zeros_like(self.image[0])
        
        mask[0:249, 0:249]      = data[self.ref_aqu[0,1]:self.ref_aqu[0,1]+249,self.ref_aqu[0,0]:self.ref_aqu[0,0]+249]
        mask[0:250, 250:499]    = data[self.ref_aqu[1,1]:self.ref_aqu[1,1]+249,self.ref_aqu[1,0]:self.ref_aqu[1,0]+249] ##BUG unclear why one goes 250 ##Not sure, but maybe because [:250] means from 1-249
        mask[0:249, 500:749]    = data[self.ref_aqu[2,1]:self.ref_aqu[2,1]+249,self.ref_aqu[2,0]:self.ref_aqu[2,0]+249]
        mask[0:249, 750:-1]     = data[self.ref_aqu[3,1]:self.ref_aqu[3,1]+249,self.ref_aqu[3,0]:self.ref_aqu[3,0]+249]
        
        self.mask               = np.tile(mask, (self.image.shape[0], 1, 1))
        
    
    def load_dark(self):
        """
        loads dark image
        """
        dark                    = fits.getdata("ACQ_dark07_20171229_DIT.fits")
        dark                    = np.nanmean(dark,0)
        
        if self.test:
            plt.title("mean of dark")
            plt.imshow(dark, origin="lower", norm=LogNorm())
            plt.show()
            
        data = np.zeros_like(self.image[0])
        data[0:249, 0:249]      = dark[self.ref_aqu[0,1]:self.ref_aqu[0,1]+249,self.ref_aqu[0,0]:self.ref_aqu[0,0]+249]
        data[0:250, 250:499]    = dark[self.ref_aqu[1,1]:self.ref_aqu[1,1]+249,self.ref_aqu[1,0]:self.ref_aqu[1,0]+249] ##BUG unclear why one goes 250 ##Not sure, but maybe because [:250] means from 1-249
        data[0:249, 500:749]    = dark[self.ref_aqu[2,1]:self.ref_aqu[2,1]+249,self.ref_aqu[2,0]:self.ref_aqu[2,0]+249]
        data[0:249, 750:-1]     = dark[self.ref_aqu[3,1]:self.ref_aqu[3,1]+249,self.ref_aqu[3,0]:self.ref_aqu[3,0]+249]
        data                    = np.tile(data, (self.image.shape[0], 1, 1))
        self.dark               = data.copy()
        
        if self.verbose:
            print("dark shape:", self.dark.shape)

        if self.test:
            plt.title("the cut dark")
            plt.imshow(np.nanmean(data,axis=0), origin="lower", norm=LogNorm())
            plt.figure()
            plt.title("Dark minus image")
            plt.imshow(np.nanmean(self.image, axis=0) - np.mean(data, axis=0), origin="lower", norm=LogNorm())

            plt.show()
        
    def get_interpolation(self):
        from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
        if self.mask is None: self.get_bad_pixelmask()
        
        mask                    = self.mask.copy()
        image                   = self.image.copy() ## image uncorrected
        image_dark              = image - self.dark ## image uncorrected, dark subtracted
        
        image_dark[mask == True]     = np.nan
        
        kernel                  = Gaussian2DKernel(y_stddev=1, x_stddev=1)
        fixed_image             = np.array([interpolate_replace_nans(i, kernel) for i in image_dark])
        
        self.image_uncorrected  = self.image.copy()
        self.image              = fixed_image.copy()
        
        if self.test:
            fig, axes = plt.subplots(1,3)
            axes[0].imshow(np.nanmean(fixed_image, axis=0), origin='lower', norm=LogNorm(vmin=10, vmax=100))
            axes[0].title.set_text("image corrected - dark")  
            
            axes[1].imshow(np.nanmean(image_dark, axis=0), origin='lower', norm=LogNorm(vmin=10, vmax=300))
            axes[1].title.set_text("image uncorrected - dark")  
            
            axes[2].imshow(np.nanmean(image, axis=0), origin='lower', norm=LogNorm(vmin=10, vmax=300))
            axes[2].title.set_text("image uncorrected")  

            plt.show()
            
            print("The dead pixels are interpolated assuming the deadpixel mask stored in the module")
            fig, axes               = plt.subplots(1,2, figsize=(16,10))
            axes[0].imshow(np.nanmedian(self.image, axis=0), origin="lower", norm=LogNorm())
            axes[1].imshow(np.nanmedian(fixed_image, axis=0), origin="lower", norm=LogNorm())
            #plt.show()
            
        del(image, mask, kernel, fixed_image)
        gc.collect()
    
    def get_aquisition_camera_plot(self, average=True, show=True):
        fig, ax                 = plt.subplots()
        if average:
            image               = np.nanmedian(self.image[:10,:,:], axis=0)
        else:
            image               = self.image[0]
        
        image[self.pos_sc[0][0], self.pos_sc[0][1]] = 15000000
        image[self.pos_sc[1][0], self.pos_sc[1][1]] = 15000000
        image[self.pos_sc[2][0], self.pos_sc[2][1]] = 15000000
        image[self.pos_sc[3][0], self.pos_sc[3][1]] = 15000000
        
        image[self.pos_ft[0][0], self.pos_ft[0][1]] = 15000000
        image[self.pos_ft[1][0], self.pos_ft[1][1]] = 15000000
        image[self.pos_ft[2][0], self.pos_ft[2][1]] = 15000000
        image[self.pos_ft[3][0], self.pos_ft[3][1]] = 15000000
        
        
        ax.imshow(image, origin="lower", norm=LogNorm(vmax=150000000))
        if show:
            plt.show()

    def get_time(self):
        time                    = np.cumsum(np.repeat(self.exptime/self.ndit, self.ndit))
        print("Use time from the P2VM reduced files, there is a aquistion table with timestamps...")
        if self.test:
            print(time)
            print(self.exptime)
        
        time_ms                 = np.array([np.timedelta64(int(ti*1000), "ms") for ti in time])
        self.timestamps         = np.array([self.date + ti for ti in time_ms])
        self.timestamps_str     = np.array([str(pd.to_datetime(ti)) for ti in self.timestamps])

class ScienceImage(AquisitionImage):
    def __init__(self, image, header, test=False, verbose=False, stack=20, correct_dead_pixels=True, sigma=3., box_size=(10,10), filter_size=(3,3)):
        """
        A bit higher level class assuming that the AquistionImage is used for Science
        args:
        image: a np.ndarray array. Test for up to 3D
        header: a fits header image_object

        kwargs:
        stack: number of single exposures that are stacked, default=20
        correct_dead_pixels: bool, if True dead pixels stored in the deadpixel mask are interpolated
        test: bool if True implented autotests are preformed
        verbose: bool if True code becomes talkative
        
        functions:
        
        self.get_science_fields()
        kwargs:
        crop_out: tuple, region to be cut out [px] default=(50,50) science fiber position +- 50 px
        
        sets attributes:
        self.raw_image      the copy of the raw image as created in AquisitionImage
        self.image          the nanmedian of all telescopes
        self.sub_images     all four telescope images
        
        self.get_sky_subtraction()
        kwargs: sigma, box_size: Box size for background estimation, filter_size: Box size to filter median background in boxes
        subtracts sky/background from self.image
        sets attributes:
            self.cube (3dim), self.cube_with_background (3dim), self.image_with_background (2dim), self.image (2dim, without sky)
        
        self.get_frames():
        gets the frames according to the number of stacks
        sets attributes:
            self.timeframes time of frames in 2019.XX-style
            
        sets attributes:
            self.frames     the frames stacked according to self.stack
            self.frame_times the respective times 
            
        self.get_frame_plot():
        simple plot helper
        
        self.save_fits():
        saves the science images to a HDUList fits file_names, naming convention is input file name + _aquisition_camera_cube.fits
        args: 
        store_dir the storage store_dir
        cube                    = fits.PrimaryHDU(self.frames, header=self.header) PrimaryHDU holds the frames
        raw                     = fits.ImageHDU(self.image) indivudal exposures
        image                   = fits.ImageHDU(np.nanmedian(self.image, axis=0)) collapse image of the file
        psf                     = fits.ImageHDU(self.psf) the psf extracted, WARNING PSF is shit, this should not be used.
        """
        
        super().__init__(image, header, test=test, verbose=verbose, correct_dead_pixels=correct_dead_pixels)
        # set attributes
        self.stack              = int(stack)
        # sky subtraction controlls
        self.sigma, self.box_size, self.filter_size = sigma, box_size, filter_size
        self.timeframes = []
        
        self.get_science_fields()
        self.get_sky_subtraction()
        self.get_frames()
        
    def get_science_fields(self, crop_out=(50, 50)):
        image                   = []
        for pos in self.pos_sc:
            image.append(self.image[:, pos[0]-50:pos[0]+50, pos[1]-50:pos[1]+50])            
        if self.test:
            for i in image:
                
                plt.figure()
                plt.imshow(np.nanmedian(i[:, :, ::-1] , axis=0), origin="lower", norm=LogNorm() )
                plt.plot(49,49, "o")
            plt.figure()
            plt.imshow(np.nanmedian(image, axis=(0,1))[:,::-1], origin="lower", norm=LogNorm())
            plt.show()
            
        self.raw_image          = self.image.copy()
        self.image              = np.nanmedian(np.array(image), axis=(0)) ###BUG this definition is different form GalacticCenterImage, where self.image is just self.raw_image --- ###WARNING not sure if thats actually true!
        self.sub_images         = np.array(image)

	def get_ft_fields(self, crop_out=(50, 50)):
        image                   = []
        	for pos in self.pos_sc:
            	image.append(self.image[:, pos[0]-50:pos[0]+50, pos[1]-50:pos[1]+50])            
        	if self.test:
            for i in image:
                
                plt.figure()
                plt.imshow(np.nanmedian(i[:, :, ::-1] , axis=0), origin="lower", norm=LogNorm() )
                plt.plot(49,49, "o")
            plt.figure()
            plt.imshow(np.nanmedian(image, axis=(0,1))[:,::-1], origin="lower", norm=LogNorm())
            plt.show()
            
        self.raw_image          = self.image.copy()
        self.image              = np.nanmedian(np.array(image), axis=(0)) ###BUG this definition is different form GalacticCenterImage, where self.image is just self.raw_image --- ###WARNING not sure if thats actually true!
        self.sub_images         = np.array(image)	
    
    def get_sky_subtraction(self, sigma=None, box_size=None, filter_size=None):
        """
        subtracts background from image
        """
        if sigma is None:
            sigma               = self.sigma
        if box_size is None:
            box_size            = self.box_size
        if filter_size is None:
            filter_size         = self.filter_size
            
        self.cube               = self.image.copy()
        image                   = self.image.copy()
        image                   = np.nanmedian(image, axis=0)
    
        sigma_clip              = SigmaClip(sigma=sigma)
        bkg_estimator           = MedianBackground()
        bkg                     = Background2D(image, box_size, filter_size=filter_size, sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)
        
        self.image_with_background = image
        b                       = image - bkg.background
        self.image              = b-b.min()
        self.cube_with_background = self.cube.copy()
        self.cube               = [(c - bkg.background) for c in self.cube]
        self.cube               = np.array([c - c.min() for c in self.cube])

        if self.verbose:    
            print('Median background: ', bkg.background_median)
            print('Median background rms: ', bkg.background_rms_median)
        
        if self.test:
            fig, axes = plt.subplots(1,3)
            b = image - bkg.background
            axes[0].imshow(self.image, origin='lower', norm=LogNorm(vmax=300))
            axes[0].title.set_text("image - background")  
            
            axes[1].imshow(image, origin='lower', norm=LogNorm(vmax=300))
            axes[1].title.set_text("image")            
            
            axes[2].imshow(bkg.background, origin='lower', norm=LogNorm(vmax=300))
            axes[2].title.set_text("background")
            
            plt.show()
    
    def get_frames(self):
        t0 = np.datetime64(self.header["HIERARCH ESO PCR ACQ START"])
        t1 = np.datetime64(self.header["HIERARCH ESO PCR ACQ END"])
        print(t0)
        print(t1)
        delta_t = (t1-t0)/self.ndit*self.stack//2
        print(delta_t)
        c                       = 0
        frames, frame_times, f0 = [], [], []
        c00                     = 0
        for index, im in enumerate(self.cube):
            c                   +=1
            if c <= self.stack:
                f0.append(im)
            else:
                frames.append(np.median(f0, axis=0))
                frame_times.append(t0 + delta_t*2*c00+ delta_t)
                c               = 0
                f0              = []
                c00             +=1
  
        self.frames             = np.array(frames)
        self.frame_times        = np.array(frame_times)
    
        for f_t in self.frame_times:
            t = AstropyTime(str(f_t)).decimalyear
            self.timeframes.append(t)
        
        del(frames, frame_times)
        gc.collect()
        
        if self.test:
            plt.plot(self.timestamps,'.')
            plt.plot(frame_times, "o")
            plt.show()
            
    def get_frame_plot(self):
        for fi, t in zip(self.frames, self.frame_times):
            plt.figure()                 
            plt.imshow(fi[:,::-1], origin="lower", norm=LogNorm(), )

        plt.show()
            
    def save_fits(self, store_dir, overwrite=False):
        name                    = store_dir + self.name[:-5] + "_aquisition_camera_cube.fits"
        
        cube                    = fits.PrimaryHDU(self.frames, header=self.header)
        raw                     = fits.ImageHDU(self.raw_image)
        image                   = fits.ImageHDU(self.image)
        try:

            ascii.write(self.sources, store_dir + self.name[:-5] + "_source_table.csv", overwrite=overwrite)
            
        except AttributeError:
            print("no sources...")
            hdul                = fits.HDUList([cube, raw, image])
            
        
        hdul                    = fits.HDUList([cube, raw, image])
        hdul.writeto(name, overwrite=overwrite)
        
class GalacticCenterImage(ScienceImage):
    def __init__(self, image, header, test=False, verbose=False, correct_dead_pixels=True):
        """
        A bit higher level class assuming that the AquistionImage is used for Science
        args:
        image: a np.ndarray array. Test for up to 3D
        header: a fits header image_object

        kwargs:
        stack: number of single exposures that are stacked, default=20
        correct_dead_pixels: bool, if True dead pixels stored in the deadpixel mask are interpolated
        test: bool if True implented autotests are preformed
        verbose: bool if True code becomes talkative
        
        functions:
        self.get_PSF()
        gets a psf, dont use this - PSF is not good
        
        self.get_stars()
        uses daophot to identify point sources in the image
        tries to get information of self.sources on some stars (S30, S65, S4, S10, S2)
        sets attributes:
            self.sources    an astropy table containg the sources found!
        
        self.compute_distance_star()
        kwarg: which (star)
        returns position in pixels of which
        
        self.get_position_Sgr()
        returns position of SgrA* based on position of S30 from self.sources and distance from self.compute_distance_star()
        
        self.get_star_names()
        kwarg: atol     tolerance for star detection of determined position and position from self.sources
        Adds name to stars in self.sources            
        
        self.get_S30(), self.get_S65(), self.get_S4(), self.get_S2(), self.get_S10():
        functions save column of these stars from self.sources
        set attributes:
            self.S30, self.S65, self.S4, self.S2, self.S10
        
        self.get_deconvolution():
        does not work
        
        """
        
        super().__init__(image, header, test=test, verbose=verbose, correct_dead_pixels=correct_dead_pixels)
        self.sources            = None
        self.get_stars()
        self.compute_distance_star()
        self.get_star_names()
        
        
    def get_PSF(self, test=None):
        if test is None: test   = self.test
        from astropy.table import Table
        from astropy.nddata import NDData
        from photutils.psf import extract_stars
        from scipy.interpolate import interp2d        
        data                    = self.image.copy()

        psf_stars_x             = [19, 54, 71, 60, 78, 74, 68, 81]
        psf_stars_y             = [69, 32, 58, 54, 50, 17, 36, 61]
        
        if test:
            plt.imshow(data, origin="lower", norm=LogNorm(vmax=100))
            plt.plot(psf_stars_x, psf_stars_y, "o", color="white")
            plt.show()
        stars_tbl               = Table()
        stars_tbl['x']          = psf_stars_x
        stars_tbl['y']          = psf_stars_y

        nddata                  = NDData(data=data)
        stars                   = extract_stars(nddata, stars_tbl, size=15)
        staro = []
        X, Y                    = np.arange(0,15), np.arange(0, 15)
        X, Y                    = np.meshgrid(X,Y)
        for star in stars:
            s                   = np.array(star)
            s                   = (s-s.min())/(s.max() + s.min())
            f                   = interp2d(X, Y, s, kind="cubic")
            x,y                 = np.linspace(0, 15, 30), np.linspace(0, 15, 30)
            staro.append(f(x, y))

        if test:
            plt.imshow(np.nanmedian(staro, axis=0), origin="lower", norm=LogNorm())
            plt.show()
        self.psf                = np.nanmedian(staro, axis=0)
        #x,y                     = np.meshgrid(x, y)
        #f                       = interp2d(x, y, self.psf, kind="cubic")
        #self.psf                = f(np.arange(0,15), np.arange(0, 15))
        
    def get_stars(self, test=None):
        if test is None: test   = self.test
        from photutils import DAOStarFinder
        data                    = self.image.copy()
        daofind                 = DAOStarFinder(fwhm=4.5, threshold=3.)
        self.sources            = daofind(data)
    
        if test:
            plt.imshow(data, origin="lower", norm=LogNorm(vmax=100))
            plt.plot(self.sources["xcentroid"], self.sources["ycentroid"], "o", color="white", alpha=0.5)
            plt.show()
            
        try:
            self.get_S30()
        except IndexError:
            self.S30 = None
        try:
            self.get_S65()
        except IndexError:
            self.S65 = None
        try:
            self.get_S2()
        except IndexError:
            self.S2 = None
        try:
            self.get_S10()
        except IndexError:
            self.S10 = None
        try:
            self.get_S4()   
        except IndexError:
            self.S4 = None
            
    def compute_distance_star(self, which='S30'):        
        """
        returns the position which in pixel for mean time of observation
        """       
        def matrix(phi):
            phi = np.radians(phi)
            return np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])
        
        time = np.mean(self.timeframes)

        x_S65 = -769.3 + 2.39*(time-2004.38)                            #[mas] Gillessen March 2017
        y_S65 = -279.1 + (-1.48)*(time-2004.38)
        x_S65_pix = x_S65 / PIXEL_SCALE                                 #position [px]
        y_S65_pix = y_S65 / PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S65_pix, y_S65_pix]))
        x_S65_pix, y_S65_pix = a[0], a[1]
        
        x_S30 = -556.9 + 1.2*(time-2004.38)                
        y_S30 = 393.9 + 3.39*(time-2004.38)
        x_S30_pix = x_S30 / PIXEL_SCALE            
        y_S30_pix = y_S30 / PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S30_pix, y_S30_pix]))
        x_S30_pix, y_S30_pix = a[0], a[1]
   
        x_S10 = 43 + (-5.04)*(time-2005.42)
        y_S10 = -374.7 + 3.04*(time-2005.42)
        x_S10_pix = x_S10 / PIXEL_SCALE          
        y_S10_pix = y_S10 / PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S10_pix, y_S10_pix]))
        x_S10_pix, y_S10_pix = a[0], a[1]
      
        if which=='S65':
            return x_S65_pix, y_S65_pix
        if which=='S30':
            return x_S30_pix, y_S30_pix
        if which=='S10':
            return x_S10_pix, y_S10_pix   
    
    def get_position_Sgr(self, which='S30'):
        """
        returns posisition of SgrA* 
        """
        try:
            x_star, y_star = self.S30['xcentroid'], self.S30['ycentroid'] 
        except TypeError:
            x_star, y_star = 14.5, 69.9
            print("warning, s30 not found using guesstimate.")
        distance = self.compute_distance_star(which=which)
        position_Sgr = (x_star - distance[0], y_star - distance[1])
        return position_Sgr
        
    def get_star_names(self, atol=3.):
        """
        detects stars and adds name to self.sources
        """
        stars = dict()
        time = np.mean(self.timeframes)
        position_Sgr = self.get_position_Sgr()
        
        xcentroid = self.sources['xcentroid']
        ycentroid = self.sources['ycentroid']
        try:
            self.sources['star']
        except KeyError:
            a = Column([None for i in range(len(xcentroid))], name='star')
            self.sources.add_column(a, index=1)
        
        def matrix(phi):
            phi = np.radians(phi)
            return np.array([[np.cos(phi), -np.sin(phi)],[np.sin(phi), np.cos(phi)]])
        
        x_S35 = (546.2 + 1.82*(time-2004.38))/PIXEL_SCALE               #[mas] Gillessen March 2017
        y_S35 = (-430.1 + 3.06*(time-2004.38))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S35, y_S35]))
        x_S35_pix, y_S35_pix = a[0], a[1]
        position_S35 = (position_Sgr[0] + x_S35_pix, position_Sgr[1] + y_S35_pix)
        stars['S35'] = position_S35
        
        x_S11 = (176.1 + 8.79*(time-2004.38))/PIXEL_SCALE
        y_S11 = (-574.9 + (-5.58)*(time-2004.38))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S11, y_S11]))
        x_S11_pix, y_S11_pix = a[0], a[1]
        position_S11 = (position_Sgr[0] + x_S11_pix, position_Sgr[1] + y_S11_pix)
        stars['S11'] = position_S11
        
        x_S7 = (516.4 + (-3.7)*(time-2004.38))/PIXEL_SCALE
        y_S7 = (-46.9 + (-2.9)*(time-2004.39))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S7, y_S7]))
        x_S7_pix, y_S7_pix = a[0], a[1]
        position_S7 = (position_Sgr[0] + x_S7_pix, position_Sgr[1] + y_S7_pix)
        stars['S7'] = position_S7
        
        x_S27 = (148.9 + 0.63*(time-2005.42))/PIXEL_SCALE
        y_S27 = (531.5 + 3.07*(time-2005.42))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S27, y_S27]))
        x_S27_pix, y_S27_pix = a[0], a[1]
        position_S27 = (position_Sgr[0] + x_S27_pix, position_Sgr[1] + y_S27_pix)
        stars['S27'] = position_S27
        
        x_S65 = (-769.3 + 2.39*(time-2004.38))/PIXEL_SCALE
        y_S65 = (-279.1 + (-1.48)*(time-2004.38))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S65, y_S65]))
        x_S65_pix, y_S65_pix = a[0], a[1]
        position_S65 = (position_Sgr[0] + x_S65_pix, position_Sgr[1] + y_S65_pix)
        stars['S65'] = position_S65
        
        x_S30 = (-556.9 + 1.2*(time-2004.38))/PIXEL_SCALE
        y_S30 = (393.9 + 3.39*(time-2004.38))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S30, y_S30]))
        x_S30_pix, y_S30_pix = a[0], a[1]
        position_S30 = (position_Sgr[0] + x_S30_pix, position_Sgr[1] + y_S30_pix)
        stars['S30'] = position_S30
   
        x_S10 = (43 + (-5.04)*(time-2005.42))/PIXEL_SCALE
        y_S10 = (-374.7 + 3.04*(time-2005.42))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S10, y_S10]))
        x_S10_pix, y_S10_pix = a[0], a[1]
        position_S10 = (position_Sgr[0] + x_S10_pix, position_Sgr[1] + y_S10_pix)
        stars['S10'] = position_S10
        
        x_S26 = (539.9 + 6.3*(time-2005.42))/PIXEL_SCALE
        y_S26 = (439.4 + 0.85*(time-2005.42))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S26, y_S26]))
        x_S26_pix, y_S26_pix = a[0], a[1]
        position_S26 = (position_Sgr[0] + x_S26_pix, position_Sgr[1] + y_S26_pix)
        stars['S26'] = position_S26
        
        x_S45 = (169.3 + (-5.58)*(time-2009.43))/PIXEL_SCALE
        y_S45 = (-538.4 + (-4.55)*(time-2009.43))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S45, y_S45]))
        x_S45_pix, y_S45_pix = a[0], a[1]
        position_S45 = (position_Sgr[0] + x_S45_pix, position_Sgr[1] + y_S45_pix)
        stars['S45'] = position_S45
        
        x_S46 = (246.5 + 0.44*(time-2005.42))/PIXEL_SCALE
        y_S46 = (-556.6 + 4.66*(time-2005.42))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S46, y_S46]))
        x_S46_pix, y_S46_pix = a[0], a[1]
        position_S46 = (position_Sgr[0] + x_S46_pix, position_Sgr[1] + y_S46_pix)
        stars['S46'] = position_S46
        
        x_S25 = (-107.6 + (-2.87)*(time-2005.42))/PIXEL_SCALE
        y_S25 = (-426.6 + 1.27*(time-2005.42))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S25, y_S25]))
        x_S25_pix, y_S25_pix = a[0], a[1]
        position_S25 = (position_Sgr[0] + x_S25_pix, position_Sgr[1] + y_S25_pix)
        stars['S25'] = position_S25
        
        x_S20 = (200.3 + (-5.04)*(time-2009.94))/PIXEL_SCALE
        y_S20 = (81.4 + (-5.63)*(time-2009.94))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S20, y_S20]))
        x_S20_pix, y_S20_pix = a[0], a[1]
        position_S20 = (position_Sgr[0] + x_S20_pix, position_Sgr[1] + y_S20_pix)
        stars['S20'] = position_S20
        
        x_S36 = (274.0 + (-0.61)*(time-2008.53))/PIXEL_SCALE
        y_S36 = (234.7 + (-1.67)*(time-2008.53))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S36, y_S36]))
        x_S36_pix, y_S36_pix = a[0], a[1]
        position_S36 = (position_Sgr[0] + x_S36_pix, position_Sgr[1] + y_S36_pix)
        stars['S36'] = position_S36
        
        x_S47 = (378.1 + (-2.84)*(time-2008.03))/PIXEL_SCALE
        y_S47 = (247.1 + 4.26*(time-2008.03))/PIXEL_SCALE
        a = np.dot(matrix(9),np.array([x_S47, y_S47]))
        x_S47_pix, y_S47_pix = a[0], a[1]
        position_S47 = (position_Sgr[0] + x_S47_pix, position_Sgr[1] + y_S47_pix)
        stars['S47'] = position_S47

        for j in stars:
            args_x_star = np.where(np.isclose(xcentroid, stars[j][0], atol=atol))[0]
            args_y_star = np.where(np.isclose(ycentroid, stars[j][1], atol=atol))[0]
            try:
                index_star = np.intersect1d(args_x_star, args_y_star)[0]
                self.sources[index_star]['star'] = j
                #print(self.sources["star"])
                #print("found a star!")
            except IndexError:
                #print("index error")
                pass
            
            #print(self.sources[index_star])
        
        if self.test:
            plt.imshow(self.image, origin='lower', norm=LogNorm(vmax=100))
            plt.plot(self.sources['xcentroid'], self.sources['ycentroid'], 'o', color='blue')
            plt.plot(self.sources["xcentroid"][self.sources["star"]!=None], self.sources["ycentroid"][self.sources["star"]!=None], 'o', color='orange')
            for name in self.sources['star'][self.sources['star']!=None]:
                x = self.sources["xcentroid"][self.sources["star"] ==  name]
                y = self.sources["ycentroid"][self.sources["star"] ==  name]
                plt.annotate(name, xy=(x,y))
            plt.show()
            
    def get_S30(self, atol=5.):
        """
        gets information of S30 from self.sources
        """
        pos_S30 = [14.5, 69.9]                                                     
                
        xcentroid = self.sources['xcentroid']
        ycentroid = self.sources['ycentroid']
        
        args_x_S30 = np.where(np.isclose(xcentroid, pos_S30[0], atol=atol))[0]
        args_y_S30 = np.where(np.isclose(ycentroid, pos_S30[1], atol=atol))[0]
        
        index_S30 = np.intersect1d(args_x_S30, args_y_S30)[0]
        self.S30 = self.sources[index_S30]
        
        if self.test:
            x_S30 = self.sources['xcentroid'][index_S30]
            y_S30 = self.sources['ycentroid'][index_S30]
            plt.imshow(self.image, norm=LogNorm(), origin='lower')
            plt.plot(x_S30, y_S30, 'o')
            plt.xlim(0,100)
            plt.ylim(0,100)
            plt.show()  

    def get_S65(self, atol=5.):
        """
        gets information of S65 from self.sources
        """
        pos_S65 = [10,26.7]                                                    
        
        xcentroid = self.sources['xcentroid']
        ycentroid = self.sources['ycentroid']

        args_x_S65 = np.where(np.isclose(xcentroid, pos_S65[0], atol=atol))[0]
        args_y_S65 = np.where(np.isclose(ycentroid, pos_S65[1], atol=atol))[0]
        
        index_S65 = np.intersect1d(args_x_S65, args_y_S65)[0]
        
        x_S65 = self.sources['xcentroid'][index_S65]
        y_S65 = self.sources['ycentroid'][index_S65]
        
        self.S65 = self.sources[index_S65]
        
        if self.test:
            plt.imshow(self.image, norm=LogNorm(), origin='lower')
            plt.plot(x_S65, y_S65, 'o')
            plt.xlim(0,100)
            plt.ylim(0,100)
            plt.show()
        
    def get_S2(self, atol=5.):
        """
        gets information of S2 from self.sources
        """
        pos_S2 = [49,54.4]
        
        xcentroid = self.sources['xcentroid']
        ycentroid = self.sources['ycentroid']
        
        args_x_S2 = np.where(np.isclose(xcentroid, pos_S2[0], atol=atol))[0]
        args_y_S2 = np.where(np.isclose(ycentroid, pos_S2[1], atol=atol))[0]
        
        index_S2 = np.intersect1d(args_x_S2, args_y_S2)[0]
        
        x_S2 = self.sources['xcentroid'][index_S2]
        y_S2 = self.sources['ycentroid'][index_S2]
        
        self.S2 = self.sources[index_S2]
        
        if self.test:
            plt.imshow(self.image, norm=LogNorm(), origin='lower')
            plt.plot(x_S2, y_S2, 'o')
            plt.xlim(0,100)
            plt.ylim(0,100)
            plt.show()
        
    def get_S10(self, atol=5.):
        """
        gets information of S10 from self.sources
        """
        pos_S10 = [51,31]
        
        xcentroid = self.sources['xcentroid']
        ycentroid = self.sources['ycentroid']
        
        args_x_S10 = np.where(np.isclose(xcentroid, pos_S10[0], atol=atol))[0]
        args_y_S10 = np.where(np.isclose(ycentroid, pos_S10[1], atol=atol))[0]
        
        index_S10 = np.intersect1d(args_x_S10, args_y_S10)[0]
        
        x_S10 = self.sources['xcentroid'][index_S10]
        y_S10 = self.sources['ycentroid'][index_S10]
        
        self.S10 = self.sources[index_S10]
        
        if self.test:
            plt.imshow(self.image, norm=LogNorm(), origin='lower')
            plt.plot(x_S10, y_S10, 'o')
            plt.xlim(0,100)
            plt.ylim(0,100)
            plt.show()
        
    def get_S4(self, atol=5.):
        """
        gets information of S4 from self.sources
        """
        pos_S4 = [66,56.5]
        
        xcentroid = self.sources['xcentroid']
        ycentroid = self.sources['ycentroid']
        
        args_x_S4 = np.where(np.isclose(xcentroid, pos_S4[0], atol=atol))[0]
        args_y_S4 = np.where(np.isclose(ycentroid, pos_S4[1], atol=atol))[0]
        
        index_S4 = np.intersect1d(args_x_S4, args_y_S4)[0]
        
        x_S4 = self.sources['xcentroid'][index_S4]
        y_S4 = self.sources['ycentroid'][index_S4]
        
        self.S4 = self.sources[index_S4]
        
        if self.test:
            plt.imshow(self.image, norm=LogNorm(), origin='lower')
            plt.plot(x_S4, y_S4, 'o')
            plt.xlim(0,100)
            plt.ylim(0,100)
            plt.show()
        
    def get_deconvolution(self):
        from scipy.signal import  convolve2d
        from skimage import restoration
        
        data                    = self.image.copy()
        
        #deconvolved_RL = restoration.unsupervised_wiener(data, self.psf)
        
        
        plt.imshow(deconvolved_RL, origin="lower")
        plt.figure()
        plt.imshow(data, origin="lower", norm=LogNorm())
        plt.show()
    

            
#opener("/home/sebastiano/Documents/Data/GRAVITY/data/")
