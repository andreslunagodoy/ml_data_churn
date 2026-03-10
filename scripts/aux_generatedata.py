from sklearn.model_selection import train_test_split

from src.config import CONFIG, BASE_DIR
from src.data_loader import load_raw_data

def main():

    INPUT_DIR = BASE_DIR / "input"

    # Load data
    df_raw = load_raw_data()

    # Split features and target
    X_raw = df_raw.drop(columns=CONFIG['target_column'])
    y = df_raw[CONFIG['target_column']].map(CONFIG['target_map'])

    # Split train and test sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=CONFIG['test_size'], stratify=y, random_state=CONFIG['random_state']
    )

    export_path = INPUT_DIR
    export_path.mkdir(parents=True, exist_ok=True)

    files = {
        "new_data.csv": X_test_raw,
        "values.csv": y_test,}

    for name, data in files.items():
        data.to_csv(export_path / name, index=False)

if __name__ == "__main__":
    main()
