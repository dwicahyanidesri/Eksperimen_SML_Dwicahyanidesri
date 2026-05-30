import os
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Direktori tempat script ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# LOAD DATASET
# ============================================================
def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Args:
        filepath (str): Path ke file CSV.

    Returns:
        pd.DataFrame: DataFrame mentah.
    """
    logger.info(f"Memuat dataset dari: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Dataset berhasil dimuat — shape: {df.shape}")
    return df


# ============================================================
# VALIDASI KOLOM
# ============================================================
REQUIRED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]


def validate_columns(df: pd.DataFrame) -> None:
    """
    Memastikan semua kolom yang dibutuhkan tersedia di dataset.

    Args:
        df (pd.DataFrame): DataFrame input.

    Raises:
        ValueError: Jika ada kolom yang tidak ditemukan.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing_cols}")
    logger.info("Validasi kolom: OK")


# ============================================================
# STEP 5.1 — MENANGANI MISSING VALUES
# ============================================================
def handle_missing_values(df: pd.DataFrame) -> tuple:
    """
    Menangani missing values pada dataset.

    Dataset tidak memiliki NaN eksplisit, namun kolom smoking_history
    mengandung nilai 'No Info' yang secara semantik merupakan implicit
    missing value. Nilai tersebut diganti dengan modus dari nilai yang
    diketahui (bukan 'No Info').

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame setelah penanganan.
            - str: Nilai modus yang digunakan sebagai pengganti.
            - int: Jumlah baris 'No Info' yang diganti.
    """
    count_no_info = (df["smoking_history"] == "No Info").sum()
    modus_smoking = (
        df[df["smoking_history"] != "No Info"]["smoking_history"].mode()[0]
    )
    df["smoking_history"] = df["smoking_history"].replace("No Info", modus_smoking)
    logger.info(
        f"[5.1] Tangani 'No Info' pada smoking_history — {count_no_info:,} baris "
        f"diganti dengan modus: '{modus_smoking}'"
    )
    return df, modus_smoking, count_no_info


# ============================================================
# STEP 5.2 — MENGHAPUS DATA DUPLIKAT
# ============================================================
def remove_duplicates(df: pd.DataFrame) -> tuple:
    """
    Menghapus baris duplikat dari dataset.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame tanpa duplikat.
            - int: Jumlah baris sebelum penghapusan.
            - int: Jumlah baris sesudah penghapusan.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    logger.info(
        f"[5.2] Hapus duplikat — sebelum: {before:,} | sesudah: {after:,} | dihapus: {removed:,}"
    )
    return df, before, after


# ============================================================
# STEP 5.3 — NORMALISASI / STANDARISASI FITUR NUMERIK
# ============================================================
def normalize_features(df: pd.DataFrame, columns: list) -> tuple:
    """
    Menerapkan StandardScaler (z-score normalization) pada fitur numerik.

    Rumus: z = (x - mean) / std

    Catatan: pada notebook eksperimen, normalisasi dilakukan pada
    seluruh dataset sebelum split. Scaler di-fit di sini untuk
    konsistensi dengan tahapan eksperimen.

    Args:
        df (pd.DataFrame): DataFrame input.
        columns (list): Kolom numerik yang akan distandarisasi.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame setelah standarisasi.
            - StandardScaler: Objek scaler yang sudah di-fit.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    logger.info(f"[5.3] StandardScaler diterapkan pada kolom: {columns}")
    return df, scaler


# ============================================================
# STEP 5.4 — DETEKSI DAN PENANGANAN OUTLIER (IQR CAPPING)
# ============================================================
def cap_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Mendeteksi dan menangani outlier menggunakan metode IQR Capping.

    Nilai di luar batas [Q1 - 1.5*IQR, Q3 + 1.5*IQR] akan di-clip
    ke batas tersebut.

    Sesuai notebook, capping diterapkan pada kolom 'bmi' setelah
    standarisasi karena outlier paling signifikan ada di kolom tersebut.

    Args:
        df (pd.DataFrame): DataFrame input.
        columns (list): Daftar kolom yang akan di-cap.

    Returns:
        pd.DataFrame: DataFrame setelah capping.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        n_outlier = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        logger.info(
            f"[5.4] Capping outlier '{col}' — {n_outlier:,} outlier | "
            f"batas: [{lower:.3f}, {upper:.3f}]"
        )
    return df


# ============================================================
# STEP 5.5 — ENCODING DATA KATEGORIKAL
# ============================================================
def encode_categorical(df: pd.DataFrame) -> tuple:
    """
    Melakukan Label Encoding pada fitur kategorikal:
    'gender' dan 'smoking_history'.

    Args:
        df (pd.DataFrame): DataFrame input.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame setelah encoding.
            - dict: Dictionary berisi objek LabelEncoder per kolom.
    """
    encoders = {}
    categorical_cols = ["gender", "smoking_history"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logger.info(f"[5.5] Label Encoding '{col}' — mapping: {mapping}")

    return df, encoders


# ============================================================
# STEP 5.6 — BINNING (PENGELOMPOKAN DATA)
# ============================================================
def add_binning_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur baru hasil binning berdasarkan nilai z-score
    (setelah standarisasi):

      age_group:
        0 = Muda         (z < -1.0)
        1 = Dewasa Awal  (-1.0 <= z < 0.0)
        2 = Dewasa Akhir (0.0  <= z < 1.0)
        3 = Lansia       (z >= 1.0)

      bmi_group:
        0 = Kurus      (z < -1.0)
        1 = Normal     (-1.0 <= z < 0.0)
        2 = Overweight (0.0  <= z < 1.0)
        3 = Obesitas   (z >= 1.0)

    Args:
        df (pd.DataFrame): DataFrame input (fitur sudah distandarisasi).

    Returns:
        pd.DataFrame: DataFrame dengan tambahan kolom age_group & bmi_group.
    """
    bin_edges = [-float("inf"), -1.0, 0.0, 1.0, float("inf")]
    bin_labels = [0, 1, 2, 3]

    # Binning usia
    age_label_map = {0: "Muda", 1: "Dewasa Awal", 2: "Dewasa Akhir", 3: "Lansia"}
    df["age_group"] = pd.cut(
        df["age"], bins=bin_edges, labels=bin_labels
    ).astype(int)
    dist_age = df["age_group"].value_counts().sort_index()
    logger.info(
        "[5.6] Binning 'age' -> 'age_group': "
        + " | ".join(f"{age_label_map[k]}={dist_age.get(k, 0):,}" for k in bin_labels)
    )

    # Binning BMI
    bmi_label_map = {0: "Kurus", 1: "Normal", 2: "Overweight", 3: "Obesitas"}
    df["bmi_group"] = pd.cut(
        df["bmi"], bins=bin_edges, labels=bin_labels
    ).astype(int)
    dist_bmi = df["bmi_group"].value_counts().sort_index()
    logger.info(
        "[5.6] Binning 'bmi' -> 'bmi_group': "
        + " | ".join(f"{bmi_label_map[k]}={dist_bmi.get(k, 0):,}" for k in bin_labels)
    )

    return df


# ============================================================
# STEP 5.7 — SPLIT DATA & SIMPAN HASIL PREPROCESSING
# ============================================================
def split_and_save(
    df: pd.DataFrame,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Membagi dataset menjadi train/test dengan stratifikasi,
    lalu menyimpan kedua subset ke folder output.

    Args:
        df (pd.DataFrame): DataFrame hasil preprocessing lengkap.
        output_dir (str): Folder tujuan penyimpanan.
        test_size (float): Proporsi data test (default 0.2).
        random_state (int): Random seed (default 42).

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f"[5.7] Split data — train: {len(X_train):,} | test: {len(X_test):,} "
        f"| rasio: {int((1 - test_size) * 100)}:{int(test_size * 100)}"
    )

    # Simpan hasil
    os.makedirs(output_dir, exist_ok=True)

    df_train = X_train.copy()
    df_train["diabetes"] = y_train.values

    df_test = X_test.copy()
    df_test["diabetes"] = y_test.values

    train_path = os.path.join(output_dir, "diabetes_train_preprocessed.csv")
    test_path = os.path.join(output_dir, "diabetes_test_preprocessed.csv")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    logger.info(f"[5.7] Data train disimpan: {train_path}  (shape: {df_train.shape})")
    logger.info(f"[5.7] Data test  disimpan: {test_path}  (shape: {df_test.shape})")

    return X_train, X_test, y_train, y_test


# ============================================================
# PIPELINE UTAMA
# ============================================================
def run_preprocessing_pipeline(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
) -> tuple:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan,
    sesuai tahapan pada notebook eksperimen Eksperimen_Dwi-Cahyani-Desri.ipynb.

    Urutan tahapan:
        5.1  Menangani Missing Values ('No Info' -> modus)
        5.2  Menghapus Data Duplikat
        5.3  Normalisasi / Standarisasi Fitur (StandardScaler)
        5.4  Deteksi dan Penanganan Outlier (IQR Capping pada 'bmi')
        5.5  Encoding Data Kategorikal (Label Encoding)
        5.6  Binning (age -> age_group, bmi -> bmi_group)
        5.7  Split Data & Simpan Hasil Preprocessing

    Args:
        input_path (str): Path ke file CSV mentah.
        output_dir (str): Folder untuk menyimpan hasil preprocessing.
        test_size (float): Proporsi data test (default 0.2).

    Returns:
        tuple: (X_train, X_test, y_train, y_test) yang siap dilatih.
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING")
    logger.info("=" * 60)

    # Load & validasi
    df = load_data(input_path)
    validate_columns(df)
    df_clean = df.copy()

    # 5.1 Missing values
    df_clean, modus_smoking, count_no_info = handle_missing_values(df_clean)

    # 5.2 Hapus duplikat
    df_clean, before, after = remove_duplicates(df_clean)

    # 5.3 Normalisasi
    numerical_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    df_clean, scaler = normalize_features(df_clean, numerical_cols)

    # 5.4 Outlier capping pada 'bmi'
    df_clean = cap_outliers_iqr(df_clean, columns=["bmi"])

    # 5.5 Encoding kategorikal
    df_clean, encoders = encode_categorical(df_clean)

    # 5.6 Binning
    df_clean = add_binning_features(df_clean)

    # 5.7 Split & simpan
    X_train, X_test, y_train, y_test = split_and_save(
        df_clean, output_dir, test_size=test_size
    )

    logger.info("=" * 60)
    logger.info("PIPELINE SELESAI — Data siap untuk pelatihan model!")
    logger.info("=" * 60)

    return X_train, X_test, y_train, y_test


# ============================================================
# ENTRY POINT (CLI)
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate preprocessing — Diabetes Prediction Dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(SCRIPT_DIR, "diabetes_prediction_raw.csv"),
        help="Path ke file CSV mentah (default: diabetes_prediction_raw.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(SCRIPT_DIR, "diabetes_preprocessing"),
        help="Folder output hasil preprocessing (default: diabetes_preprocessing/)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data test antara 0.0–1.0 (default: 0.2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    X_train, X_test, y_train, y_test = run_preprocessing_pipeline(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
    )

    print()
    print("=" * 60)
    print("RINGKASAN HASIL PREPROCESSING")
    print("=" * 60)
    print(f"  X_train shape  : {X_train.shape}")
    print(f"  X_test  shape  : {X_test.shape}")
    print(f"  Kolom fitur    : {list(X_train.columns)}")
    print(f"  y_train dist   : {dict(y_train.value_counts().sort_index())}")
    print(f"  y_test  dist   : {dict(y_test.value_counts().sort_index())}")
    print(f"  Output folder  : {args.output}/")
    print("=" * 60)

