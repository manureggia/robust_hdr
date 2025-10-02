import argparse
import os
import numpy as np

from robust_hdr.utils import get_metadata, imread_nef, save_exr, save_tonemapped_png, build_estimate_mask, list_nef
from robust_hdr.exposure_ratio import estimate_exposure_mst
from robust_hdr.hdr_merge import demosaic_merge

def main():
    parser = argparse.ArgumentParser(description="Tool Minimale per la creazione di immagini HDR partendo da diverse immagini SDR attravreso l'algoritmo GreedyMST")
    parser.add_argument("dir", help="Cartella con i RAW (.NEF)")
    parser.add_argument("-t", "--tiles", type=int, default=16, help="Numero di tasselli (num_tiles x num_tiles)")
    parser.add_argument("--noise-floor", type=int, default=16, help="Parametro sotto il quale è considerato rumore")
    parser.add_argument("-l", "--lambda-reg", type=float, default=1e-2, help="Termine di regolarizzazione di Tikonov"), 
    parser.add_argument("-o", "--out", default="preview.png", help="PNG tonemappato")
    parser.add_argument("--exr", default=None, help="Salva l'immagine in formato OpenEXR")
    parser.add_argument("--exr-dtype", default="float32", choices=["half", "float32"])
    args = parser.parse_args()

    # Semplificazione: Ordinamento Alfabetico = Ordinamento per Esposizione
    nef_files = list_nef(args.dir)
    if not nef_files:
        raise SystemExit(f"Nessun .NEF trovato in {args.dir}")

    md = get_metadata(nef_files)

    Y = np.stack([imread_nef(f, md, libraw=False) for f in nef_files], axis=0)  # (N,H,W) in [0,1]

    estimate, Y, black_frame = build_estimate_mask(Y, md)

    # Stima dei tempi (se ho abbastanza scatti “buoni”)
    if estimate.sum() > 2:
        res = estimate_exposure_mst(
            Y[estimate],
            md['exp'][estimate],
            md,
            lambda_reg=args.lambda_reg,
            num_tiles = args.tiles,
            noise_floor= args.noise_floor
        )
        # La funzione può dare solo exp oppure (exp, loss): prendo sempre exp
        exp_hat = res[0] if isinstance(res, tuple) else res
        exp_hat = np.asarray(exp_hat, dtype=md['exp'].dtype)
        md['exp'][estimate] = exp_hat

    # Merge RAW 
    HDR, num_sat = demosaic_merge(nef_files, md)

    # niente valori negativi
    if HDR.min() < 0:
        HDR = np.clip(HDR, 0, None)

    # Preview PNG
    save_tonemapped_png(args.out, HDR)
    print(f"Preview tonemappata salvata in {args.out}")

    # EXR opzionale
    if args.exr:
        save_exr(args.exr, HDR.astype(np.float32), dtype=args.exr_dtype)
        print(f"EXR salvato in {args.exr} ({args.exr_dtype})")


if __name__ == "__main__":
    main()
