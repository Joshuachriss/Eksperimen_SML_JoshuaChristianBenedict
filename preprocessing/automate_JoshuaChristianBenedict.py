# automate_iris.py
# File otomatisasi preprocessing untuk dataset Iris
# Kriteria 1 - Skilled/Advance

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath: Path ke file CSV

    Returns:
        DataFrame hasil load
    """
    df = pd.read_csv(filepath)
    logger.info(f"Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    logger.info(f"Kolom: {df.columns.tolist()}")
    return df


def drop_unnecessary_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Menghapus kolom yang tidak diperlukan untuk pemodelan.

    Args:
        df: Input DataFrame
        cols: Daftar kolom yang akan dihapus

    Returns:
        DataFrame setelah kolom dihapus
    """
    existing_cols = [c for c in cols if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
        logger.info(f"Kolom dihapus: {existing_cols}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values dengan menghapus baris yang memiliki nilai kosong.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame tanpa missing values
    """
    before = len(df)
    df = df.dropna()
    after = len(df)
    removed = before - after
    if removed > 0:
        logger.info(f"Missing values: {removed} baris dihapus")
    else:
        logger.info("Tidak ada missing values ditemukan")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame tanpa duplikat
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    if removed > 0:
        logger.info(f"Duplikat: {removed} baris dihapus")
    else:
        logger.info("Tidak ada duplikat ditemukan")
    return df


def handle_outliers_iqr(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Menangani outlier menggunakan metode IQR Clipping.
    Nilai di luar batas [Q1 - 1.5*IQR, Q3 + 1.5*IQR] di-clip.

    Args:
        df: Input DataFrame
        feature_cols: Daftar kolom fitur numerik

    Returns:
        DataFrame setelah outlier di-clip
    """
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        if n_outliers > 0:
            logger.info(f"Outlier clipping [{col}]: {n_outliers} nilai di-clip")
    logger.info("Penanganan outlier (IQR clipping) selesai")
    return df


def encode_target(df: pd.DataFrame, target_col: str = 'species') -> tuple:
    """
    Melakukan Label Encoding pada kolom target kategorikal.

    Args:
        df: Input DataFrame
        target_col: Nama kolom target

    Returns:
        Tuple (DataFrame, LabelEncoder)
    """
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    mapping = dict(enumerate(le.classes_))
    logger.info(f"Label encoding selesai. Mapping: {mapping}")
    return df, le


def scale_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Standarisasi fitur numerik menggunakan StandardScaler (mean=0, std=1).

    Args:
        df: Input DataFrame
        feature_cols: Daftar kolom fitur numerik

    Returns:
        Tuple (DataFrame, StandardScaler)
    """
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    logger.info(f"Standarisasi selesai untuk kolom: {feature_cols}")
    return df, scaler


def save_preprocessed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Menyimpan DataFrame hasil preprocessing ke file CSV.

    Args:
        df: DataFrame yang akan disimpan
        output_path: Path tujuan penyimpanan
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Data preprocessing disimpan di: {output_path}")


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Pipeline preprocessing lengkap untuk dataset Iris.

    Tahapan:
        1. Load data
        2. Drop kolom tidak relevan (Id)
        3. Tangani missing values
        4. Hapus duplikat
        5. Tangani outlier (IQR clipping)
        6. Label encoding target
        7. Standarisasi fitur

    Args:
        input_path: Path ke file CSV raw
        output_path: Path tujuan file CSV hasil preprocessing

    Returns:
        DataFrame hasil preprocessing
    """
    logger.info("=" * 50)
    logger.info("MEMULAI PIPELINE PREPROCESSING IRIS")
    logger.info("=" * 50)

    # Kolom fitur
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    target_col = 'species'

    df = load_data(input_path)

    df = drop_unnecessary_columns(df, cols=['Id'])

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = handle_outliers_iqr(df, feature_cols)

    df, le = encode_target(df, target_col)

    df, scaler = scale_features(df, feature_cols)

    save_preprocessed_data(df, output_path)

    logger.info("=" * 50)
    logger.info(f"PREPROCESSING SELESAI | Shape final: {df.shape}")
    logger.info("=" * 50)

    return df


if __name__ == "__main__":
    INPUT_PATH = "iris_raw/iris.csv"
    OUTPUT_PATH = "preprocessing/iris_preprocessing/iris_preprocessing.csv"

    result = preprocess(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH
    )
    print(f"\nPreview data hasil preprocessing:")
    print(result.head())
    print(f"\nShape: {result.shape}")
