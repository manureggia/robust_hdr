# robust-hdr

Tool minimale per la creazione di immagini HDR partendo da una serie di immagini SDR (RAW/NEF).
Utilizza l'algoritmo MST (Maximum Spanning Tree) di Hanji & Mantiuk (2023) per stimare in modo robusto i rapporti di
esposizione, invece di fidarsi unicamente dei metadati EXIF che spesso risultano imprecisi.  
La stima è resa più stabile con regolarizzazione di Tikhonov. Una volta stimati i tempi, le immagini vengono fuse in HDR lineare tramite demosaic + merge pesato, con possibilità di salvataggio in EXR o anteprima PNG tonemappata.

## Installazione

Clonare il repository e installare con pip in modalità locale:

```bash
pip install .
```

Richiede Python 3.9+.

## Utilizzo da riga di comando

Dalla cartella del progetto (dove sono i file NEF):

```bash
robust-hdr ./cartella_con_raw \
  --lambda-reg 1e-2 \
  --out preview.png \
  --exr output.exr
```

Opzioni principali:
- `--lambda-reg`: peso della regolarizzazione (default `1e-2`)
- `--out`: file PNG tonemappato di anteprima
- `--exr`: salva il risultato in formato HDR OpenEXR
- `--exr-dtype`: precisione EXR (`half` o `float32`)

## Utilizzo da Python

```python
from robust_hdr import estimate_exposure_mst, demosaic_merge, save_exr
import numpy as np

# files = lista di RAW/NEF
# md = get_metadata(files)

Y = np.stack([imread_nef(f, md, libraw=False) for f in files], axis=0)
exp = estimate_exposure_mst(Y, md['exp'], md, lambda_reg=1e-2)
HDR, mask = demosaic_merge(files, md)

save_exr("risultato.exr", HDR.astype(np.float32))
```

## Dipendenze principali
- numpy
- rawpy
- Pillow
- exifread
- openexr + imath
- tqdm