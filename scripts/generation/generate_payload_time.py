#!/usr/bin/env python3
"""Generate payload_length and time_diff using pretrained NeCSTGen models.

This helper loads the pretrained VAE and GMM models and generates packets for
each flow present in the input CSV. Only ``payload_length`` and ``time_diff``
features are saved in the output.
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib


def load_models(proto: str, models_dir: str):
    """Load VAE encoder/decoder and GMM for a protocol."""
    enc = tf.keras.models.load_model(
        f"{models_dir}/VAE/encoder_vae_MONDAY_T11_{proto}_FINAL.h5",
        custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU},
    )
    dec = tf.keras.models.load_model(
        f"{models_dir}/VAE/decoder_vae_MONDAY_T11_{proto}_FINAL.h5",
        custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU},
    )
    gmm = joblib.load(
        f"{models_dir}/GMM/gmm_MONDAY_BL100_T11_{proto}_FLOWS_FINAL.sav"
    )
    return enc, dec, gmm


FEATURES = {
    "HTTP": [
        "layers_2",
        "layers_3",
        "layers_4",
        "layers_5",
        "flags",
        "sport",
        "dport",
        "length_total",
        "time_diff",
        "rate",
        "rolling_rate_byte_sec",
        "rolling_rate_byte_min",
        "rolling_rate_packet_sec",
        "rolling_rate_packet_min",
        "header_length",
        "payload_length",
    ],
    "UDP_GOOGLE_HOME": [
        "layers_2",
        "layers_3",
        "layers_4",
        "layers_5",
        "length_total",
        "time_diff",
        "rate",
        "rolling_rate_byte_sec",
        "rolling_rate_byte_min",
        "rolling_rate_packet_sec",
        "rolling_rate_packet_min",
        "header_length",
        "payload_length",
    ],
}

# indexes of features in processed data
IDX_TIME_DIFF = {"HTTP": 8, "UDP_GOOGLE_HOME": 5}
IDX_PAYLOAD = {"HTTP": 15, "UDP_GOOGLE_HOME": 12}


def scale_back(
    x: np.ndarray, col: str, df_raw: pd.DataFrame, log_scale: bool = False
) -> np.ndarray:
    """Reverse the normalization applied during training.

    Parameters
    ----------
    x : np.ndarray
        Normalized values to scale back.
    col : str
        Column name in ``df_raw`` used to recover min and max.
    df_raw : pd.DataFrame
        Raw dataframe containing the original values.
    log_scale : bool, optional
        If ``True`` apply a base-10 exponentiation after rescaling to
        undo the log10 transform used during training. Defaults to ``False``.
    """

    scaled = x * (df_raw[col].max() - df_raw[col].min()) + df_raw[col].min()
    return np.power(10, scaled) if log_scale else scaled


def generate_packets(dec, gmm, n_packets: int) -> np.ndarray:
    """Sample latent vectors from the GMM and decode packets."""
    z, _ = gmm.sample(n_packets)
    return dec.predict(z)


def main():
    parser = argparse.ArgumentParser(description="Generate packet features")
    parser.add_argument("--protocol", choices=["HTTP", "UDP_GOOGLE_HOME"], required=True)
    parser.add_argument("--input", help="CSV file containing a flow_id column", required=True)
    parser.add_argument("--output", help="Path to save generated CSV", required=True)
    parser.add_argument("--models-dir", default="models", help="Directory with pretrained models")
    args = parser.parse_args()

    enc, dec, gmm = load_models(args.protocol, args.models_dir)

    df_counts = pd.read_csv(args.input)
    if "flow_id" not in df_counts.columns:
        raise ValueError("input file must contain a flow_id column")
    flow_sizes = df_counts.groupby("flow_id").size()

    df_raw = pd.read_csv(f"data/raw/df_raw_{args.protocol}.csv")

    results = []
    for fid, count in flow_sizes.items():
        feats = generate_packets(dec, gmm, int(count))
        cols = FEATURES[args.protocol]
        df_feat = pd.DataFrame(feats, columns=cols)
        payl = scale_back(
            df_feat.iloc[:, IDX_PAYLOAD[args.protocol]].to_numpy(),
            "payload_length",
            df_raw,
            log_scale=False,
        )
        time = scale_back(
            df_feat.iloc[:, IDX_TIME_DIFF[args.protocol]].to_numpy(),
            "time_diff",
            df_raw,
            log_scale=True,
        )
        tmp = pd.DataFrame({"flow_id": fid, "payload_length": payl, "time_diff": time})
        results.append(tmp)

    pd.concat(results, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
