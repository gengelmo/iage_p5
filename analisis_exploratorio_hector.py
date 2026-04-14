import os
from pathlib import Path

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

	object_columns = df.select_dtypes(include="object")
	if not object_columns.empty:
		print("\nCadenas vacías por columna:")
		print(object_columns.eq("").sum().to_string())
	else:
		print("\nNo hay columnas de texto para evaluar cadenas vacías.")

	print("\nValores duplicados:")
	print(df.duplicated().sum())

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

	if "Symbol" in df.columns:
		print("\nNúmero de símbolos únicos:")
		print(df["Symbol"].nunique())


def main() -> None:
	dataframe, overall_date_range, file_date_ranges = load_stock_files()
	if dataframe.empty:
		print("\nNo se ha podido construir un dataframe con datos válidos.")
		return

	exploratory_analysis(dataframe, overall_date_range, file_date_ranges)


if __name__ == "__main__":
	main()
