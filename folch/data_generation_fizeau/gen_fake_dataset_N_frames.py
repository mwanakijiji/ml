'''
Generate four fake datasets of LBT Fizeau PSFs, each a sequence of frames
driven by random walks in OPD, tip, and/or tilt:

    opd_only      random-walk OPD (std ~10 um), tip = tilt = 0
    tip_only      random-walk tip (std 0.05 asec), OPD = tilt = 0
    tilt_only     random-walk tilt (std 0.05 asec), OPD = tip = 0
    opd_tip_tilt  independent random walks in all three

Each frame is written as its own FITS file (cropped to 60x60 to match the
autoencoder input), with OPD (m), TIP and TILT (asec) in the header:

    fake_datasets/<dataset_name>/frame_000.fits, ...
'''

import os

import numpy as np
import matplotlib
matplotlib.use("Agg") # write plots to file; no windows
import matplotlib.pyplot as plt
from astropy.io import fits

from simulate_LBT_PSF_mwe import make_intensity, crop_center


# number of frames per dataset
N_FRAMES = 100

# target scatter (std over the whole walk)
OPD_SCATTER = 10e-6 # m
TIP_SCATTER = 0.05 # arcsec
TILT_SCATTER = 0.05 # arcsec

# optics / detector
WAVEL = 3.7e-6 # m
N_PIX = 2048 # pixels across the image of the aperture
D = 8.25 # m
PS_LMIR = 0.0107 # arcsec/pixel on detector

# half-width of the cropped frame (60x60 matches the autoencoder input)
CROP_HALF = 30

SEED = 42

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_datasets")


def random_walk(n, scatter, rng):
    '''
    Random walk of n values, mean-subtracted and rescaled so its std equals scatter.

    INPUTS:
    n (int): number of values
    scatter (float): target standard deviation of the walk
    rng (np.random.Generator): random number generator

    OUTPUTS:
    walk (array): random walk with mean 0 and std = scatter
    '''

    walk = np.cumsum(rng.normal(loc=0.0, scale=1.0, size=n))

    return (walk - np.mean(walk)) / np.std(walk) * scatter


def write_dataset(name, opd, tip, tilt):
    '''
    Generate and write one dataset of PSF frames as FITS files.

    INPUTS:
    name (str): name of the dataset (subdirectory of OUT_DIR)
    opd (array): OPD of each frame, in meters
    tip (array): tip of each frame, in arcseconds
    tilt (array): tilt of each frame, in arcseconds
    '''

    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(len(opd)):
        intensity, _ = make_intensity(opd[i], tip[i], tilt[i],
                                      wavel=WAVEL, n_pix=N_PIX, d=D, ps_lmir=PS_LMIR)
        frame = crop_center(intensity, half=CROP_HALF)

        hdr = fits.Header()
        hdr["OPD"] = (float(opd[i]), "OPD [m]")
        hdr["TIP"] = (float(tip[i]), "tip [arcsec]")
        hdr["TILT"] = (float(tilt[i]), "tilt [arcsec]")
        hdr["WAVEL"] = (WAVEL, "wavelength [m]")
        hdr["PLTSCL"] = (PS_LMIR, "plate scale [arcsec/pix]")
        hdr["FRAME"] = (i, "frame number in sequence")

        fits.writeto(os.path.join(out_dir, "frame_{:03d}.fits".format(i)),
                     frame.astype(np.float32), header=hdr, overwrite=True)

        if (i + 1) % 10 == 0:
            print("  {}: {}/{} frames".format(name, i + 1, len(opd)))

    # quick-look plot of the walks
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    axes[0].plot(1e6 * opd, marker=".")
    axes[0].set_ylabel("OPD (μm)")
    axes[1].plot(tip, marker=".")
    axes[1].set_ylabel("tip (asec)")
    axes[2].plot(tilt, marker=".")
    axes[2].set_ylabel("tilt (asec)")
    axes[2].set_xlabel("frame")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "walks.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":

    rng = np.random.default_rng(seed=SEED)
    zeros = np.zeros(N_FRAMES)

    datasets = {
        "opd_only": (random_walk(N_FRAMES, OPD_SCATTER, rng), zeros, zeros),
        "tip_only": (zeros, random_walk(N_FRAMES, TIP_SCATTER, rng), zeros),
        "tilt_only": (zeros, zeros, random_walk(N_FRAMES, TILT_SCATTER, rng)),
        "opd_tip_tilt": (random_walk(N_FRAMES, OPD_SCATTER, rng),
                         random_walk(N_FRAMES, TIP_SCATTER, rng),
                         random_walk(N_FRAMES, TILT_SCATTER, rng)),
    }

    for name, (opd, tip, tilt) in datasets.items():
        print("Generating {} ...".format(name))
        write_dataset(name, opd, tip, tilt)

    print("Done. Datasets written to {}".format(OUT_DIR))
