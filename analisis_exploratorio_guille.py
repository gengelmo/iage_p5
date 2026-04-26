# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


os.chdir(Path(__file__).resolve().parent)


def load_stock_files() -> tuple[pd.DataFrame, tuple[pd.Timestamp | None, pd.Timestamp | None], dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]]]:
	script_dir = Path.cwd()
	stocks_dir = script_dir / "data" / "Stocks"
	tmp_dir = script_dir / "tmp"
	tmp_dir.mkdir(parents=True, exist_ok=True)

	txt_files = sorted(stocks_dir.glob("*.txt"))
	if not txt_files:
		print(f"No se han encontrado archivos TXT en {stocks_dir}")
		return pd.DataFrame(), (None, None), {}

	expected_columns = None
	valid_frames = []
	incoherent_files = []
	no_data_files = []
	file_date_ranges: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = {}
	overall_min_date = None
	overall_max_date = None

	for file_path in txt_files:
		print(f"Leyendo {file_path.name}...")
		try:
			frame = pd.read_csv(file_path)
		except EmptyDataError:
			print(f"  -> {file_path.name} no tiene datos.")
			no_data_files.append(file_path.name)
			continue
		except Exception as exc:
			print(f"  -> {file_path.name} no se ha podido leer: {exc}")
			incoherent_files.append(file_path.name)
			continue

		if frame.empty:
			print(f"  -> {file_path.name} no tiene datos.")
			no_data_files.append(file_path.name)
			continue

		current_columns = frame.columns.tolist()
		if expected_columns is None:
			expected_columns = current_columns
		elif current_columns != expected_columns:
			print(
				f"  -> {file_path.name} tiene columnas incoherentes: {current_columns}"
			)
			incoherent_files.append(file_path.name)
			continue

		if "Date" in frame.columns:
			frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
			file_min_date = frame["Date"].min()
			file_max_date = frame["Date"].max()
		else:
			file_min_date = None
			file_max_date = None

		file_date_ranges[file_path.name] = (file_min_date, file_max_date)
		if file_min_date is not None:
			overall_min_date = file_min_date if overall_min_date is None else min(overall_min_date, file_min_date)
		if file_max_date is not None:
			overall_max_date = file_max_date if overall_max_date is None else max(overall_max_date, file_max_date)

		frame["Symbol"] = file_path.name.removesuffix(".us.txt")
		valid_frames.append(frame)

	if incoherent_files:
		incoherent_path = tmp_dir / "archivos_incoherentes.txt"
		incoherent_path.write_text("\n".join(incoherent_files) + "\n", encoding="utf-8")
		print(f"\nSe han guardado {len(incoherent_files)} archivos incoherentes en {incoherent_path}")

	if no_data_files:
		print(f"\nArchivos sin datos detectados: {len(no_data_files)}")

	if not valid_frames:
		return pd.DataFrame(), (None, None), file_date_ranges

	combined = pd.concat(valid_frames, ignore_index=True)

	numeric_columns = ["Open", "High", "Low", "Close", "Volume", "OpenInt"]
	for column in numeric_columns:
		if column in combined.columns:
			combined[column] = pd.to_numeric(combined[column], errors="coerce")

	return combined, (overall_min_date, overall_max_date), file_date_ranges


def clean_market_data_pandas(df: pd.DataFrame) -> pd.DataFrame:
	"""Replicate the minimal market-data cleaning used in the Spark pipeline."""
	df = df[
		(df["Open"] > 0) &
		(df["High"] > 0) &
		(df["Low"] > 0) &
		(df["Close"] > 0) &
		(df["Volume"] >= 0)
	].copy()

	df = df[
		(df["High"] >= df["Open"]) &
		(df["High"] >= df["Close"]) &
		(df["High"] >= df["Low"]) &
		(df["Low"] <= df["Open"]) &
		(df["Low"] <= df["Close"])
	].copy()

	return df


def filter_symbols_by_min_rows_pandas(df: pd.DataFrame, min_rows: int = 60) -> pd.DataFrame:
	"""Keep only symbols with at least min_rows observations."""
	counts = df.groupby("Symbol").size()
	valid_symbols = counts[counts >= min_rows].index
	return df[df["Symbol"].isin(valid_symbols)].copy()


def add_target_variable_pandas(df: pd.DataFrame) -> pd.DataFrame:
	"""Add LogReturn20 target, consistent with the Spark pipeline."""
	df = df.sort_values(["Symbol", "Date"]).copy()
	future_close = df.groupby("Symbol")["Close"].shift(-20)
	df["LogReturn20"] = np.log(future_close / df["Close"])
	return df


def split_by_date_pandas(
	df: pd.DataFrame,
	train_end: str = "2015-01-01",
	val_end: str = "2016-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Create temporal train/validation/test splits."""
	train_end_ts = pd.Timestamp(train_end)
	val_end_ts = pd.Timestamp(val_end)

	train_df = df[df["Date"] < train_end_ts].copy()
	val_df = df[(df["Date"] >= train_end_ts) & (df["Date"] < val_end_ts)].copy()
	test_df = df[df["Date"] >= val_end_ts].copy()

	return train_df, val_df, test_df


def exploratory_analysis(
	df: pd.DataFrame,
	overall_date_range: tuple[pd.Timestamp | None, pd.Timestamp | None],
	file_date_ranges: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]],
) -> None:
	print("\n==================== EDA ====================")
	print(f"Shape: {df.shape}")
	print(f"Columnas: {list(df.columns)}")

	print("\nHead:")
	print(df.head().to_string())

	print("\nTail:")
	print(df.tail().to_string())

	print("\nInformación general:")
	df.info()

	print("\nNulos por columna:")
	print(df.isna().sum().to_string())

	object_columns = df.select_dtypes(include=["object", "string"])
	if not object_columns.empty:
		print("\nCadenas vacías por columna:")
		print(object_columns.eq("").sum().to_string())
	else:
		print("\nNo hay columnas de texto para evaluar cadenas vacías.")

	print("\nValores duplicados:")
	print(df.duplicated().sum())

	if {"Symbol", "Date"}.issubset(df.columns):
		print("\nDuplicados por (Symbol, Date):")
		print(df.duplicated(subset=["Symbol", "Date"]).sum())

	overall_min_date, overall_max_date = overall_date_range
	if overall_min_date is not None and overall_max_date is not None:
		print("\nRango temporal:")
		print(f"  - Valor mínimo de la columna Date: {overall_min_date}")
		print(f"  - Valor máximo de la columna Date: {overall_max_date}")
		print(f"  - Fechas nulas tras parseo: {df['Date'].isna().sum()}")

		file_start_dates = [
			(name, dates[0])
			for name, dates in file_date_ranges.items()
			if dates[0] is not None
		]
		file_end_dates = [
			(name, dates[1])
			for name, dates in file_date_ranges.items()
			if dates[1] is not None
		]

		file_start_dates.sort(key=lambda item: item[1])
		file_end_dates.sort(key=lambda item: item[1])

		print("\nTop 10 archivos que antes empiezan:")
		for name, start_date in file_start_dates[:10]:
			print(f"  - {name}: {start_date.date()}")

		print("\nTop 10 archivos que más tarde empiezan:")
		for name, start_date in file_start_dates[-10:][::-1]:
			print(f"  - {name}: {start_date.date()}")

		end_month_year = [date.strftime("%Y-%m") for _, date in file_end_dates]
		end_counts = pd.Series(end_month_year).value_counts().sort_values(ascending=False)
		print("\nConteo de archivos por mes-año de finalización:")
		for period, count in end_counts.items():
			print(f"  - {period}: {count}")

	else:
		print("\nNo hay fechas válidas en los archivos para calcular el rango temporal.")

	print("\nResumen estadístico:")
	with pd.option_context("display.max_columns", None, "display.width", 200):
		print(df.describe(include="all").transpose().to_string())

	if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
		print("\nValores inválidos en columnas de mercado:")
		invalid_checks = {
			"Open <= 0": (df["Open"] <= 0).sum(),
			"High <= 0": (df["High"] <= 0).sum(),
			"Low <= 0": (df["Low"] <= 0).sum(),
			"Close <= 0": (df["Close"] <= 0).sum(),
			"Volume < 0": (df["Volume"] < 0).sum(),
			"Volume == 0": (df["Volume"] == 0).sum(),
		}
		for name, count in invalid_checks.items():
			print(f"  - {name}: {count} ({count / len(df):.4%})")

		print("\nInconsistencias OHLC:")
		ohlc_checks = {
			"High < Open": (df["High"] < df["Open"]).sum(),
			"High < Close": (df["High"] < df["Close"]).sum(),
			"High < Low": (df["High"] < df["Low"]).sum(),
			"Low > Open": (df["Low"] > df["Open"]).sum(),
			"Low > Close": (df["Low"] > df["Close"]).sum(),
		}
		for name, count in ohlc_checks.items():
			print(f"  - {name}: {count} ({count / len(df):.4%})")

		print("\nCuantiles de columnas numéricas:")
		quantiles = df[["Open", "High", "Low", "Close", "Volume"]].quantile(
			[0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
		)
		print(quantiles.to_string())

	if "Symbol" in df.columns:
		print("\nNúmero de símbolos únicos:")
		print(df["Symbol"].nunique())

		counts_per_symbol = df.groupby("Symbol").size()

		print("\nFilas por símbolo:")
		print(counts_per_symbol.describe().to_string())

		print("\nTop 10 símbolos con más filas:")
		print(counts_per_symbol.sort_values(ascending=False).head(10).to_string())

		print("\nSímbolos con pocas filas:")
		print(f"  - Menos de 30 filas: {(counts_per_symbol < 30).sum()}")
		print(f"  - Menos de 40 filas: {(counts_per_symbol < 40).sum()}")
		print(f"  - Menos de 60 filas: {(counts_per_symbol < 60).sum()}")

	if {"Symbol", "Date"}.issubset(df.columns):
		symbol_ranges = df.groupby("Symbol")["Date"].agg(["min", "max", "count"])
		symbol_ranges["span_days"] = (symbol_ranges["max"] - symbol_ranges["min"]).dt.days

		print("\nDuración temporal por símbolo (en días):")
		print(symbol_ranges["span_days"].describe().to_string())

	if "Date" in df.columns:
		print("\nNúmero de filas por año:")
		rows_per_year = df["Date"].dt.year.value_counts().sort_index()
		print(rows_per_year.to_string())

	if {"Date", "Symbol"}.issubset(df.columns):
		print("\nSímbolos activos por año:")
		active_symbols_per_year = df.groupby(df["Date"].dt.year)["Symbol"].nunique()
		print(active_symbols_per_year.to_string())


def analyze_label_distribution_by_split(
	df: pd.DataFrame,
	train_end: str = "2015-01-01",
	val_end: str = "2016-01-01",
	min_rows: int = 60,
) -> None:
	print("\n==================== LABEL DISTRIBUTION BY SPLIT ====================")

	work_df = clean_market_data_pandas(df)
	work_df = filter_symbols_by_min_rows_pandas(work_df, min_rows=min_rows)
	work_df = add_target_variable_pandas(work_df)
	work_df = work_df.dropna(subset=["LogReturn20"]).copy()

	train_df, val_df, test_df = split_by_date_pandas(
		work_df,
		train_end=train_end,
		val_end=val_end,
	)

	splits = {
		"train": train_df,
		"validation": val_df,
		"test": test_df,
	}

	for split_name, split_df in splits.items():
		y = split_df["LogReturn20"].dropna()

		print(f"\n--- {split_name.upper()} ---")
		print(f"Tamaño: {len(y)}")

		if len(y) == 0:
			print("Sin observaciones.")
			continue

		stats = y.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
		print(stats.to_string())

		print(f"Media: {y.mean():.6f}")
		print(f"Desviación típica: {y.std():.6f}")
		print(f"Asimetría: {y.skew():.6f}")
		print(f"Curtosis: {y.kurtosis():.6f}")


def summarize_label_distribution_table(
	df: pd.DataFrame,
	train_end: str = "2015-01-01",
	val_end: str = "2016-01-01",
	min_rows: int = 60,
) -> None:
	work_df = clean_market_data_pandas(df)
	work_df = filter_symbols_by_min_rows_pandas(work_df, min_rows=min_rows)
	work_df = add_target_variable_pandas(work_df)
	work_df = work_df.dropna(subset=["LogReturn20"]).copy()

	train_df, val_df, test_df = split_by_date_pandas(
		work_df,
		train_end=train_end,
		val_end=val_end,
	)

	rows = []
	for split_name, split_df in {
		"train": train_df,
		"validation": val_df,
		"test": test_df,
	}.items():
		y = split_df["LogReturn20"].dropna()
		rows.append({
			"split": split_name,
			"count": len(y),
			"mean": y.mean(),
			"std": y.std(),
			"min": y.min(),
			"p01": y.quantile(0.01),
			"p05": y.quantile(0.05),
			"median": y.median(),
			"p95": y.quantile(0.95),
			"p99": y.quantile(0.99),
			"max": y.max(),
		})

	summary_df = pd.DataFrame(rows)
	print("\nResumen comparativo de LogReturn20 por split:")
	print(summary_df.to_string(index=False))
	
def analyze_symbol_coverage_by_split(
	df: pd.DataFrame,
	train_end: str = "2015-01-01",
	val_end: str = "2016-01-01",
	min_rows: int = 60,
) -> None:
	print("\n==================== SYMBOL COVERAGE BY SPLIT ====================")

	work_df = clean_market_data_pandas(df)
	work_df = filter_symbols_by_min_rows_pandas(work_df, min_rows=min_rows)
	work_df = add_target_variable_pandas(work_df)
	work_df = work_df.dropna(subset=["LogReturn20"]).copy()

	train_df, val_df, test_df = split_by_date_pandas(
		work_df,
		train_end=train_end,
		val_end=val_end,
	)

	train_symbols = set(train_df["Symbol"].unique())
	val_symbols = set(val_df["Symbol"].unique())
	test_symbols = set(test_df["Symbol"].unique())

	print("\nSímbolos únicos por split:")
	print(f"  - Train: {len(train_symbols)}")
	print(f"  - Validation: {len(val_symbols)}")
	print(f"  - Test: {len(test_symbols)}")

	print("\nSolapamiento de símbolos:")
	print(f"  - Validation ∩ Train: {len(val_symbols & train_symbols)}")
	print(f"  - Test ∩ Train: {len(test_symbols & train_symbols)}")
	print(f"  - Test ∩ Validation: {len(test_symbols & val_symbols)}")

	val_new_symbols = val_symbols - train_symbols
	test_new_symbols = test_symbols - train_symbols

	print("\nSímbolos no vistos en train:")
	print(f"  - Nuevos en validation: {len(val_new_symbols)}")
	print(f"  - Nuevos en test: {len(test_new_symbols)}")

	if len(val_symbols) > 0:
		print(f"  - % símbolos de validation ya vistos en train: {len(val_symbols & train_symbols) / len(val_symbols):.2%}")
	if len(test_symbols) > 0:
		print(f"  - % símbolos de test ya vistos en train: {len(test_symbols & train_symbols) / len(test_symbols):.2%}")

	val_rows_seen = val_df["Symbol"].isin(train_symbols).sum()
	test_rows_seen = test_df["Symbol"].isin(train_symbols).sum()

	print("\nCobertura por filas:")
	print(f"  - Filas de validation con símbolos ya vistos en train: {val_rows_seen} / {len(val_df)} ({val_rows_seen / len(val_df):.2%})")
	print(f"  - Filas de test con símbolos ya vistos en train: {test_rows_seen} / {len(test_df)} ({test_rows_seen / len(test_df):.2%})")

	train_counts = train_df.groupby("Symbol").size()
	val_counts = val_df.groupby("Symbol").size()
	test_counts = test_df.groupby("Symbol").size()

	print("\nDistribución de filas por símbolo dentro de cada split:")
	print("\nTrain:")
	print(train_counts.describe().to_string())

	print("\nValidation:")
	print(val_counts.describe().to_string())

	print("\nTest:")
	print(test_counts.describe().to_string())

	print("\nTop 10 símbolos con más filas en validation:")
	print(val_counts.sort_values(ascending=False).head(10).to_string())

	print("\nTop 10 símbolos con más filas en test:")
	print(test_counts.sort_values(ascending=False).head(10).to_string())

	train_history_thresholds = [60, 252, 756]

	print("\nCobertura de validation/test según historial previo en train:")
	for threshold in train_history_thresholds:
		well_known_symbols = set(train_counts[train_counts >= threshold].index)

		val_rows_well_known = val_df["Symbol"].isin(well_known_symbols).sum()
		test_rows_well_known = test_df["Symbol"].isin(well_known_symbols).sum()

		print(f"\n  Umbral: {threshold} filas previas en train")
		print(f"    - Validation cubierto: {val_rows_well_known} / {len(val_df)} ({val_rows_well_known / len(val_df):.2%})")
		print(f"    - Test cubierto: {test_rows_well_known} / {len(test_df)} ({test_rows_well_known / len(test_df):.2%})")

def analyze_eligible_symbols_for_symbol_feature(
	df: pd.DataFrame,
	train_end: str = "2015-01-01",
	val_end: str = "2016-01-01",
	min_rows: int = 60,
	train_min: int = 252,
	val_min: int = 20,
	test_min: int = 20,
) -> None:
	print("\n==================== ELIGIBLE SYMBOLS FOR USING SYMBOL FEATURE ====================")

	work_df = clean_market_data_pandas(df)
	work_df = filter_symbols_by_min_rows_pandas(work_df, min_rows=min_rows)
	work_df = add_target_variable_pandas(work_df)
	work_df = work_df.dropna(subset=["LogReturn20"]).copy()

	train_df, val_df, test_df = split_by_date_pandas(
		work_df,
		train_end=train_end,
		val_end=val_end,
	)

	train_counts = train_df.groupby("Symbol").size().rename("train_rows")
	val_counts = val_df.groupby("Symbol").size().rename("val_rows")
	test_counts = test_df.groupby("Symbol").size().rename("test_rows")

	symbol_split_counts = pd.concat(
		[train_counts, val_counts, test_counts],
		axis=1
	).fillna(0)

	symbol_split_counts = symbol_split_counts.astype(int)
	symbol_split_counts["total_rows"] = (
		symbol_split_counts["train_rows"] +
		symbol_split_counts["val_rows"] +
		symbol_split_counts["test_rows"]
	)

	eligible_symbols = symbol_split_counts[
		(symbol_split_counts["train_rows"] >= train_min) &
		(symbol_split_counts["val_rows"] >= val_min) &
		(symbol_split_counts["test_rows"] >= test_min)
	].copy()

	print("\nCriterios de elegibilidad:")
	print(f"  - train >= {train_min}")
	print(f"  - validation >= {val_min}")
	print(f"  - test >= {test_min}")

	print(f"\nSímbolos elegibles: {len(eligible_symbols)}")

	if len(symbol_split_counts) > 0:
		print(
			f"Porcentaje sobre todos los símbolos tras limpieza: "
			f"{len(eligible_symbols) / len(symbol_split_counts):.2%}"
		)

	if not eligible_symbols.empty:
		print("\nResumen de símbolos elegibles:")
		print(eligible_symbols.describe().to_string())

		print("\nTop 10 símbolos elegibles con más filas totales:")
		print(
			eligible_symbols
			.sort_values("total_rows", ascending=False)
			.head(10)
			.to_string()
		)

		print("\nTop 10 símbolos elegibles con menos filas en train:")
		print(
			eligible_symbols
			.sort_values(["train_rows", "val_rows", "test_rows"], ascending=True)
			.head(10)
			.to_string()
		)

		tmp_dir = Path.cwd() / "tmp"
		tmp_dir.mkdir(parents=True, exist_ok=True)

		output_path = tmp_dir / "eligible_symbols_for_symbol_feature.csv"
		eligible_symbols.reset_index().to_csv(output_path, index=False)

		print(f"\nListado guardado en: {output_path}")
	else:
		print("\nNo hay símbolos que cumplan esos criterios.")
		
def analyze_binary_target_balance_by_split(
    df: pd.DataFrame,
    train_end: str = "2015-01-01",
    val_end: str = "2016-01-01",
    min_rows: int = 60,
) -> None:
    print("\n==================== BINARY TARGET BALANCE BY SPLIT ====================")

    work_df = clean_market_data_pandas(df)
    work_df = filter_symbols_by_min_rows_pandas(work_df, min_rows=min_rows)
    work_df = add_target_variable_pandas(work_df)
    work_df = work_df.dropna(subset=["LogReturn20"]).copy()

    work_df["target_bin"] = (work_df["LogReturn20"] > 0).astype(int)

    train_df, val_df, test_df = split_by_date_pandas(
        work_df,
        train_end=train_end,
        val_end=val_end,
    )

    summary_rows = []

    for split_name, split_df in {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }.items():
        total = len(split_df)
        positives = int((split_df["target_bin"] == 1).sum())
        negatives = int((split_df["target_bin"] == 0).sum())

        pos_pct = positives / total if total > 0 else 0.0
        neg_pct = negatives / total if total > 0 else 0.0
        majority_class = 1 if positives >= negatives else 0

        print(f"\n--- {split_name.upper()} ---")
        print(f"Total: {total}")
        print(f"Positivos (1): {positives} ({pos_pct:.2%})")
        print(f"Negativos (0): {negatives} ({neg_pct:.2%})")
        print(f"Clase mayoritaria: {majority_class}")

        summary_rows.append({
            "split": split_name,
            "total": total,
            "positives": positives,
            "negatives": negatives,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "majority_class": majority_class,
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\nResumen comparativo del target binario por split:")
    print(summary_df.to_string(index=False))

def main() -> None:
	dataframe, overall_date_range, file_date_ranges = load_stock_files()
	if dataframe.empty:
		print("\nNo se ha podido construir un dataframe con datos válidos.")
		return

	exploratory_analysis(dataframe, overall_date_range, file_date_ranges)
	analyze_label_distribution_by_split(dataframe)
	summarize_label_distribution_table(dataframe)
	analyze_symbol_coverage_by_split(dataframe)
	analyze_eligible_symbols_for_symbol_feature(
		dataframe,
		train_min=252,
		val_min=60,
		test_min=60,
	)
	analyze_binary_target_balance_by_split(dataframe)
	

# %%
if __name__ == "__main__":
	main()