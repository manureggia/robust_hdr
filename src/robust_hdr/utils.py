import os
import numpy as np
from PIL import Image
from fractions import Fraction


import rawpy
import OpenEXR, Imath
import exifread


# -------------------------- METADATA --------------------------

def get_metadata(files, color_space='sRGB', sat_percent=0.98, black_level=0, bits=None):
    """
    Estrae metadati necessari per stima/merge:
      - exp: tempi di esposizione (s) da EXIF
      - gain: ISO normalizzato (ISO/100)
      - aperture: area di apertura (diagnostico)
      - h,w: dimensioni immagine (postprocess default)
      - black_level: per canale (array)
      - white_level: massimo RAW
      - saturation_point: white_level * sat_percent
      - white_balance: WB camera (R,G,B)
      - black_level_norm, saturation_norm: soglie normalizzate in [0,1]
    """
    data = dict()
    data['N'] = len(files)

    # --- EXIF ---
    try:
        data['exp'], data['gain'], data['aperture'] = np.empty((3, data['N']))
        for i, file in enumerate(files):
            with open(file, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            # ExposureTime
            if 'EXIF ExposureTime' in tags:
                data['exp'][i] = float(Fraction(tags['EXIF ExposureTime'].printable))
            elif 'Image ExposureTime' in tags:
                data['exp'][i] = float(Fraction(tags['Image ExposureTime'].printable))
            else:
                raise Exception(f'Impossibile leggere il tempo di esposizione dal file: {file}')
            # ISO
            if 'EXIF ISOSpeedRatings' in tags:
                data['gain'][i] = float(str(tags['EXIF ISOSpeedRatings'])) / 100.0
            elif 'Image ISOSpeedRatings' in tags:
                data['gain'][i] = float(str(tags['Image ISOSpeedRatings'])) / 100.0
            else:
                data['gain'][i] = 1.0
            # Aperture area (diagnostico)
            try:
                focal_length = float(Fraction(tags['EXIF FocalLength'].printable))
                f_number = float(Fraction(tags['EXIF FNumber'].printable))
                data['aperture'][i] = np.pi * (focal_length / (2.0 * f_number))**2
            except Exception:
                data['aperture'][i] = 1.0
    except Exception as e:
        raise Exception('Impossibile estrarre i metadati dai file.') from e

    # --- RAW info dal primo file ---
    try:
        data['raw_format'] = True
        with rawpy.imread(files[0]) as raw:
            rgb = raw.postprocess(user_flip=0)
            data['h'], data['w'] = rgb.shape[:2]
            bl = np.array(raw.black_level_per_channel, dtype=np.float32)
            data['black_level'] = bl
            wl = float(raw.white_level)
            data['white_level'] = wl
            data['saturation_point'] = wl * float(sat_percent)
            wb = raw.camera_whitebalance
            if wb is not None:
                data['white_balance'] = np.asarray(wb, dtype=np.float32)[:3]
            else:
                data['white_balance'] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

            longest = int(np.argmax(data['exp'] * data['gain'] * data['aperture']))
        with rawpy.imread(files[longest]) as rawL:
            raw_img = rawL.raw_image
            maxv = int(raw_img.max())
            if bits is None:
                bits = int(np.ceil(np.log2(max(1, maxv))))
            data['bits'] = bits
            data['libraw_scale'] = lambda img: img/(2**bits-1)*(2**16-1)
    except rawpy._rawpy.LibRawFileUnsupportedError as e:
        raise Exception('Formato non supportato') from e

    # Soglie normalizzate (in [0,1])
    wl = max(1.0, float(data.get('white_level', 1.0)))
    data['black_level_norm'] = float(np.mean(data.get('black_level', 0))) / wl
    data['saturation_norm']  = float(data.get('saturation_point', wl)) / wl
    data['color_space'] = color_space.lower()
    return data


# --------------------------  helper --------------------------

def list_nef(dir_path: str):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".nef")]
    files.sort()
    return files


def build_estimate_mask(Y: np.ndarray, md: dict) -> np.ndarray:
    """Segna gli scatti da usare per la stima esposizioni.
    Regola empirica: se >95% dei pixel è saturo o sotto soglia, salta lo scatto.
    """
    N = md['N']
    estimate = np.ones(N, dtype=bool)

    # black level per pattern Bayer 2x2; se RGB presente, replica su 3 canali
    black_frame = np.tile(md['black_level'].reshape(2, 2), (md['h']//2, md['w']//2))
    if Y.ndim == 4:
        assert Y.shape[-1] == 3 or Y[..., 3].all() == 0
        Y = Y[..., :3]
        black_frame = np.ones_like(Y[0]) * md['black_level'][:3][None, None]

    noise_floor = max(md['saturation_point']/1000, float(np.abs((Y[0] - black_frame).min())))

    for i in range(N):
        over = (Y[i] >= md['saturation_point']).sum() > 0.95 * Y[i].size
        under = (Y[i] - black_frame <= noise_floor).sum() > 0.95 * Y[i].size
        if over or under:
            estimate[i] = False
    return estimate, Y, black_frame




# -------------------------- I/O RAW (NEF) --------------------------

def imread_nef(file, color_space='srgb', libraw=True, wb=None):
    raw = rawpy.imread(file)
    if libraw:
        # Normalize color_space argument and map to rawpy enums
        cs = (color_space or 'srgb').lower()
        if cs == 'srgb':
            out_cs = rawpy.ColorSpace.sRGB
        elif cs == 'raw':
            out_cs = rawpy.ColorSpace.raw
        else:
            raise Exception('Colorspace non riconosciuto, utilizzare RAW o SRGB.')

        wb_kwargs = {"use_camera_wb": True} if wb is None else {"user_wb": wb}

        img16 = raw.postprocess(
            gamma=(1.0, 1.0),          # LINEARE
            no_auto_bright=True,       # niente auto exposure
            output_bps=16,             # 16-bit per canale
            user_flip=0,
            output_color=out_cs,
            **wb_kwargs
        )
        # Convert to float32 [0,1]
        img = (img16.astype(np.float32) / 65535.0)
    else:
        # Raw mosaic/visible data (nessun demosaic/postprocess). Manteniamo il tipo originale.
        img = raw.raw_image_visible
    return img



# -------------------------- SAVE FUNCTIONS --------------------------

_DEF_EPS = 1e-8

def _linear_to_srgb(y: np.ndarray) -> np.ndarray:
    """Converti da lineare a sRGB (gamma standard). y atteso in [0, +inf)."""
    y = np.clip(y, 0.0, None)
    a = 0.055
    return np.where(y <= 0.0031308, 12.92*y, (1 + a)*np.power(y, 1/2.4) - a)


def save_tonemapped_png(
    path: str,
    img_lin: np.ndarray,
    *,
    apply_reinhard: bool = True,
    key: float = 0.18,
    auto_exposure: bool = True,
    exposure_bias_ev: float = 0.0,
) -> None:

    x = img_lin.astype(np.float32)

    if auto_exposure:
        w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        L = np.clip((x * w[None, None, :]).sum(axis=2), 0.0, None)
        L_avg = np.exp(np.mean(np.log(np.maximum(L, _DEF_EPS))))
        scale = key / max(L_avg, _DEF_EPS)
        if exposure_bias_ev:
            scale *= float(2.0 ** exposure_bias_ev)
        x = x * scale

    if apply_reinhard:
        # Reinhard globale: y = x / (1 + x)
        x = x / (1.0 + np.maximum(x, 0.0))
    x = np.clip(x, 0.0, 1.0)
    srgb = _linear_to_srgb(x)
    img8 = (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8).save(path)


def save_exr(path: str, img: np.ndarray, *, dtype: str = "float32") -> None:
    """
    Salva un'immagine HDR in formato OpenEXR.
    - img: HxWx3 float (valori lineari). Supporta canali R,G,B.
    - dtype: "half" (float16) o "float32".
    """
    # Sanitize: rimuovi NaN/Inf e assicurati float32 per consistenza
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("Atteso array HxWx3 (almeno 3 canali).")
    if OpenEXR is None or Imath is None:
        raise RuntimeError("Modulo OpenEXR/Imath non disponibile. Installa con 'pip install openexr imath'.")

    H, W, _ = img.shape
    header = OpenEXR.Header(W, H)
    if dtype.lower() == "half":
        pt = Imath.PixelType(Imath.PixelType.HALF)
        cast = np.float16
    else:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        cast = np.float32

    header["channels"] = {c: Imath.Channel(pt) for c in ("R", "G", "B")}
    out_path = path if path.lower().endswith(".exr") else path + ".exr"
    out = OpenEXR.OutputFile(out_path, header)
    data = {
        "R": img[..., 0].astype(cast).tobytes(),
        "G": img[..., 1].astype(cast).tobytes(),
        "B": img[..., 2].astype(cast).tobytes(),
    }
    out.writePixels(data)
    out.close()
