import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from glob import glob
from datetime import datetime
import gc
import pandas as pd


from scipy.signal import correlate2d, convolve2d
from scipy.ndimage import shift

from matplotlib.colors import LogNorm

def opener(path):

    files                       = []
    file_names                  = []
    for fi in glob(path + "GRAVI.*fits"):
        print(fi)
        if "aqu" not in fi:
            f0                      = fits.open(fi) 
            file_names.append(fi)
            files.append(f0)
    headers                     = []
    aquistion_images            = []
    names                       = []
    for fi, finame in zip(files, file_names):
        print(finame)
        h0                      = fi[0].header 
        i0                      = fi[4].data
        n0                      = fi[0].header["ARCFILE"]
        
        if "S2" in h0["ESO INS SOBJ NAME"] and "SKY" not in h0["ESO DPR TYPE"] and len(finame) == 34:
            headers.append(h0)
            aquistion_images.append(i0)
            names.append(n0)
    
    return aquistion_images, headers


class ObservationNight(list):
    def __init__(self, path, gc=True, savedir=None):
        self.path               = path
        self.gc                 = gc
        self.image_objects      = None
        
        if savedir is None:
            self.savedir        = path
        else:
            self.savedir        = savedir
        
        self.aquistion_images, self.headers = opener(self.path)
    
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
    
    def save(self):
        for io in self.image_objects:
            io.save_fits(self.savedir)

        nightcube               = []
        for io in self.image_objects:
            nightcube.append(np.nanmedian(io.image, axis=0))
        self.nightcube          = np.array(nightcube)
        self.shifted_cube       = self.shiftcube()
        
        fits.writeto(self.savedir + "aquistion_nightcube.fits", np.array(nightcube))
        fits.writeto(self.savedir + "aquistion_shifted_nightcube.fits", np.array(self.shifted_cube))
        
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
    
    
class Image():
    def __init__(self, image, test=False, verbose=False):
        self.image              = image
        self.test, self.verbose = test, verbose
        
class AquisitionImage(Image):
    def __init__(self, image, header, test=False, verbose=False, correct_dead_pixels=True):   
        
        super().__init__(image, test=test, verbose=verbose)
        self.raw_image          = self.image.copy()
        self.header             = header
        
        self.date               = np.datetime64(datetime.strptime(self.header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S"))
        self.mjd                = self.header["MJD-OBS"]
        self.exptime            = self.header["EXPTIME"]
        self.name               = self.header["ARCFILE"]
        self.ndit               = self.header["ESO DET1 NDIT"]
        self.dit                = self.header["ESO DET1 SEQ1 DIT"]
        
    
        
        self.get_image()
        self.get_fiber()
        self.get_sc_pos()
        self.get_ft_pos()
        self.get_time()
        self.load_bad_pixelmask()
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
        
        

        m0                      = rms(self.image)
        if self. test:
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
        mask[0:250, 250:499]    = data[self.ref_aqu[1,1]:self.ref_aqu[1,1]+249,self.ref_aqu[1,0]:self.ref_aqu[1,0]+249]
        mask[0:249, 500:749]    = data[self.ref_aqu[2,1]:self.ref_aqu[2,1]+249,self.ref_aqu[2,0]:self.ref_aqu[2,0]+249]
        mask[0:249, 750:-1]       = data[self.ref_aqu[3,1]:self.ref_aqu[3,1]+249,self.ref_aqu[3,0]:self.ref_aqu[3,0]+249]
        
        self.mask               = np.tile(mask, (self.image.shape[0], 1, 1))
        
    def get_interpolation(self):
        from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
        if self.mask is None: self.get_bad_pixelmask()
        
        mask                    = self.mask.copy()
        image                   = self.image.copy()
        
        image[mask == True]     = np.nan
        kernel                  = Gaussian2DKernel(y_stddev=1, x_stddev=1)
        fixed_image             = np.array([interpolate_replace_nans(i, kernel) for i in image])
        
        self.image_uncorrected  = self.image.copy()
        self.image              = fixed_image.copy()
        
        
        if self.test:
            fig, axes               = plt.subplots(1,2)
            axes[0].imshow(np.nanmedian(self.image, axis=0), origin="lower", norm=LogNorm())
            axes[1].imshow(np.nanmedian(fixed_image, axis=0), origin="lower", norm=LogNorm())
            plt.show()
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
        time                    = np.cumsum(np.repeat(self.dit, self.ndit))
        time                    = np.array([np.timedelta64(int(ti*1000), "ms") for ti in time])
        self.timestamps         = np.array([self.date + ti for ti in time])
        self.timestamps_str     = np.array([str(pd.to_datetime(ti)) for ti in self.timestamps])

class ScienceImage(AquisitionImage):
    def __init__(self, image, header, test=False, verbose=False, stack=20, correct_dead_pixels=True):
        super().__init__(image, header, test=test, verbose=verbose, correct_dead_pixels=correct_dead_pixels)
        self.stack              = int(stack)
        self.get_science_fields()
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
        self.image              = np.nanmedian(image, axis=(0)) ###BUG this definition is different form GalacticCenterImage, where self.image is just self.raw_image 
        self.sub_images         = np.array(image)
    
    
    def get_frames(self):
        ### BUG this implementation does not take the average time between and beging but the nearst index, lazyness.
        c                       = 0
        frames, frame_times, f0 = [], [], []
        for index, i in enumerate(self.image):
            c                   +=1
            if c <= self.stack:
                f0.append(i)
            else:
                frames.append(np.median(f0, axis=0))
                frame_times.append(self.timestamps[index - self.stack//2 -1])
                c               = 0
                
        self.frames             = np.array(frames)
        self.frame_times        = np.array(frame_times)
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
        raw                     = fits.ImageHDU(self.image)
        image                   = fits.ImageHDU(np.nanmedian(self.image, axis=0))
        
        try:
            psf                 = fits.ImageHDU(self.psf)
            hdul                = fits.HDUList([cube, raw, image, psf])
        except AttributeError:
            print("no psf found")
            hdul                = fits.HDUList([cube, raw, image])
            
        
        hdul                    = fits.HDUList([cube, raw, image, psf])
        hdul.writeto(name, overwrite=overwrite)
        
class GalacticCenterImage(ScienceImage):
    def __init__(self, image, header, test=False, verbose=False, correct_dead_pixels=True):
        super().__init__(image, header, test=test, verbose=verbose, correct_dead_pixels=correct_dead_pixels)
        self.sources            = None
        
    def get_PSF(self, test=None):
        if test is None: test   = self.test
        from astropy.table import Table
        from astropy.nddata import NDData
        from photutils.psf import extract_stars
        from scipy.interpolate import interp2d        
        data                    = np.nanmedian(self.image, axis=0)

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
        data                    = np.nanmedian(self.image, axis=0)
        daofind                 = DAOStarFinder(fwhm=3.0, threshold=3.)
        self.sources            = daofind(data)
        
        
        if test:
            plt.imshow(data, origin="lower", norm=LogNorm(vmax=100))
            plt.plot(self.sources["xcentroid"], self.sources["ycentroid"], "o", color="white", alpha=0.5)
            plt.show()
            
            

        
    def get_deconvolution(self):
        from scipy.signal import  convolve2d
        from skimage import restoration
        
        data                    = np.nanmedian(self.image, axis=0)
        
        #deconvolved_RL = restoration.unsupervised_wiener(data, self.psf)
        
        
        plt.imshow(deconvolved_RL, origin="lower")
        plt.figure()
        plt.imshow(data, origin="lower", norm=LogNorm())
        plt.show()
    

            
#opener("/home/sebastiano/Documents/Data/GRAVITY/data/")
