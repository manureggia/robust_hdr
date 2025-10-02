import numpy as np


def estimate_exposure_mst(imgs, exif_exp, metadata, lambda_reg: float = 1e-2, num_tiles = 16, noise_floor = 16):
    """
    Stima i tempi di esposizione relativi da uno stack di immagini con approccio MST
    e risoluzione pesata (WLS).

    Parametri aggiuntivi:
    - lambda_reg (float): coefficiente di regolarizzazione di Tikhonov
    - num_tiles: Numero di tiles per suddividere l'immagine
    - noise_floor: Valore sotto il quale è considerato rumore
    """
    num_exp = len(imgs)
    num_msts = 50

    # Livello di nero per-pixel
    black_frame = np.tile(metadata['black_level'].reshape(2, 2), (metadata['h'] // 2, metadata['w'] // 2))
    if imgs.ndim == 4:
        assert imgs.shape[-1] == 3 or imgs[..., 3].all() == 0
        black_frame = np.ones_like(imgs[0]) * metadata['black_level'][:3][None, None]

    # Segnale utile e sotto-campionamento (canale verde)
    Y = np.maximum(imgs - black_frame, 1e-6)
    black_frame = black_frame[::2, 1::2]
    Y = Y[:, ::2, 1::2]

    # Pesi ~ 1/varianza
    scaled_var = 1 / Y

    # Crop a multipli della griglia dei tasselli
    _, h, w = Y.shape
    h = num_tiles * (h // num_tiles)
    w = num_tiles * (w // num_tiles)

    # Rimappa in tasselli (N, Ty, Tx, hy, wx)
    Y = (
        Y[:, :h, :w]
        .reshape(num_exp, num_tiles, h // num_tiles, num_tiles, w // num_tiles)
        .transpose(0, 1, 3, 2, 4)
        .astype(np.float32)
    )
    scaled_var = (
        scaled_var[:, :h, :w]
        .reshape(num_exp, num_tiles, h // num_tiles, num_tiles, w // num_tiles)
        .transpose(0, 1, 3, 2, 4)
        .astype(np.float32)
    )
    black_frame = (
        black_frame[:h, :w]
        .reshape(num_tiles, h // num_tiles, num_tiles, w // num_tiles)
        .transpose(0, 2, 1, 3)
        .astype(np.float32)
    )

    # Scarto saturi/bui
    Y[Y + black_frame >= metadata['saturation_point']] = -1
    Y[Y <= noise_floor] = -1

    # Appiattisci pixel per tassello
    Y = Y.reshape(num_exp, num_tiles * num_tiles, -1)
    scaled_var = scaled_var.reshape(num_exp, num_tiles * num_tiles, -1)

    # Seleziono campioni affidabili per tassello
    thresholds = np.sort(Y[1:])[..., -num_msts]
    valid = (Y[1:] > thresholds[..., None]) & (Y[:-1] > -1)
    num_selected = valid.sum(axis=-1).min(axis=0)

    skip = num_selected < num_msts * 0.5
    valid &= ~skip[None, :, None]

    # Costruzione di  W, O, m tassello per tassello
    W, O, m = [], [], []
    for tt in range(num_tiles * num_tiles):
        if skip[tt]:
            continue

        rows = num_selected[tt] * (num_exp - 1)
        O_tile = np.zeros((rows, num_exp), dtype=np.float32)
        W_tile, m_tile = [], []

        for ee in range(num_exp - 1):
            # Pesi preliminari
            w_pre = 1 / (scaled_var[ee, tt, valid[ee, tt]] + scaled_var[ee + 1, tt, valid[ee, tt]])
            idx = np.argsort(w_pre)[-num_selected[tt]:]

            # Si sceglie l'esposizione più lunga valida
            longest = np.zeros_like(idx)
            for ff in range(num_exp - 1, ee, -1):
                cand = Y[ff, tt, valid[ee, tt]][idx]
                new_mask = (longest == 0) & (cand > -1)
                longest[new_mask] = ff
                if (longest > 0).all():
                    break

            # Pesi definitivi e misura log‑rapporto
            w_def = 1 / (
                scaled_var[ee, tt, valid[ee, tt]][idx]
                + scaled_var[:, tt, valid[ee, tt]][:, idx][longest, np.arange(num_selected[tt])]
            )
            W_tile.append(w_def)

            m_tile.append(
                np.log(
                    Y[:, tt, valid[ee, tt]][:, idx][longest, np.arange(num_selected[tt])] / Y[ee, tt, valid[ee, tt]][idx]
                )
            )

            # -e_ee + e_longest = m
            start = ee * num_selected[tt]
            stop = (ee + 1) * num_selected[tt]
            O_tile[start:stop, ee] = -1
            O_tile[start:stop][np.arange(num_selected[tt]), longest] = 1

        W_tile, m_tile = (np.concatenate(W_tile), np.concatenate(m_tile))

        # Controllo outlier a livello di tassello contro EXIF
        e_tile = np.linalg.lstsq(np.diag(np.sqrt(W_tile)) @ O_tile, np.sqrt(W_tile) * m_tile, rcond=None)[0]
        e_tile = np.exp(e_tile - e_tile.max()) * exif_exp.max()
        if (np.abs(e_tile - exif_exp) / exif_exp > 1).any():
            continue

        O.append(O_tile)
        W.append(W_tile)
        m.append(m_tile)

    if not W:
        return exif_exp, np.nan

    W, O, m = (np.concatenate(W), np.concatenate(O), np.concatenate(m))

    # Sistema WLS globale
    A = np.diag(np.sqrt(W)) @ O
    b = np.sqrt(W) * m

    # Regolarizzazione di Tikhonov: concatena √(λ)I e √(λ)e0
    if lambda_reg and lambda_reg > 0:
        e0 = np.log(np.clip(exif_exp, 1e-12, None)).astype(A.dtype)
        I = np.eye(O.shape[1], dtype=A.dtype)
        A = np.vstack([A, np.sqrt(lambda_reg) * I])
        b = np.concatenate([b, np.sqrt(lambda_reg) * e0])

    e = np.linalg.lstsq(A, b, rcond=None)[0]
    exp = np.exp(e - e.max()) * exif_exp.max()

    reject = np.abs(exp - exif_exp) / exif_exp > 2
    exp[reject] = exif_exp[reject]
    return exp