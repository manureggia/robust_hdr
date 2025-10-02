import numpy as np
import rawpy
import tqdm
from .utils import imread_nef


def demosaic_merge(files, metadata, sat_percent=0.98):
    # Trova la più corta esposizione (meno rischio saturazione)
    exp_array = np.array(metadata['exp']) * np.array(metadata['gain']) * np.array(metadata['aperture'])
    shortest_exposure = np.argmin(exp_array)
    num_sat = 0
    num, denom = np.zeros((2, metadata['h'], metadata['w'], 3))
    for i, f in enumerate(tqdm.tqdm(files, leave=True, desc="Merging")):
        raw = rawpy.imread(f)
        img = imread_nef(f, color_space=metadata['color_space'])
        if np.issubdtype(img.dtype, np.floating):
            saturation_point_img = float(sat_percent)
        else:
            # fallback per immagini intere
            saturation_point_img = sat_percent * (np.iinfo(img.dtype).max)
        # Solo la più corta esposizione contribuisce anche se satura
        if i == shortest_exposure:
            unsaturated = np.ones_like(img, dtype=bool)
            # calcola quanti pixel saturi
            num_sat = np.count_nonzero(~unsaturated_mask(raw.raw_image_visible, metadata['saturation_point'],
                                                         img, saturation_point_img)) / 3
        else:
            unsaturated = unsaturated_mask(raw.raw_image_visible, metadata['saturation_point'],
                                           img, saturation_point_img)
        X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
        denom[unsaturated] += metadata['exp'][i]
        num[unsaturated] += X_times_t[unsaturated]
    HDR = num / denom
    return HDR, num_sat

# Maschera pixel non saturi (RAW e RGB) - adattato da HDRutils
def unsaturated_mask(raw=None, saturation_threshold=None, img=None, saturation_threshold_img=None):
    # restituisce maschera booleana pixel non saturi (True = ok)
    # Se RAW presente, controlla tutti i 4 pixel della matrice Bayer
    if raw is not None:
        unsat = np.logical_and.reduce((
            raw[0::2,0::2] < saturation_threshold,
            raw[1::2,0::2] < saturation_threshold,
            raw[0::2,1::2] < saturation_threshold,
            raw[1::2,1::2] < saturation_threshold
        ))
        # upsampling 2x per tornare alla dimensione dell'immagine RGB
        unsat4 = np.zeros([unsat.shape[0]*2, unsat.shape[1]*2], dtype=bool)
        unsat4[0::2,0::2] = unsat
        unsat4[1::2,0::2] = unsat
        unsat4[0::2,1::2] = unsat
        unsat4[1::2,1::2] = unsat
        if img is None:
            return unsat4
    # Se c'è anche l'immagine RGB, controllo saturazione dopo white balance
    assert img is not None, "Serve almeno img RGB"
    unsat_rgb = np.all(img < saturation_threshold_img, axis=2)
    if raw is None:
        return np.repeat(unsat_rgb[:, :, np.newaxis], 3, axis=-1)
    else:
        unsat4 = np.logical_and(unsat4, unsat_rgb)
        return np.repeat(unsat4[:, :, np.newaxis], 3, axis=-1)
