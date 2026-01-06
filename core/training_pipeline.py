import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TrainingPipeline:
    def __init__(
        self,
        csv_file,
        model_output,
        n_estimators=100,
        max_depth=None,
        test_size=0.2,
        random_state=42,
    ):
        self.csv_file = csv_file
        self.model_output = model_output
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state

    # ======================================================
    # DATA
    # ======================================================
    def load_data(self):
        import pandas as pd

        df = pd.read_csv(self.csv_file)

        X = df.drop(columns=["label", "target"], errors="ignore")
        y = df["label"] if "label" in df else df["target"]

        return X.values, y.values

    # ======================================================
    # TRAIN
    # ======================================================
    def train(self, progress_callback=None):
        """
        progress_callback(step, total)
        """

        # ---------- STEP 1: LOAD ----------
        if progress_callback:
            progress_callback(1, 5)

        X, y = self.load_data()

        # ---------- STEP 2: SPLIT ----------
        if progress_callback:
            progress_callback(2, 5)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # ---------- STEP 3: MODEL ----------
        if progress_callback:
            progress_callback(3, 5)

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=-1,
            random_state=self.random_state,
        )

        model.fit(X_train, y_train)

        # ---------- STEP 4: EVAL ----------
        if progress_callback:
            progress_callback(4, 5)

        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Accuracy: {acc:.3f}")

        # ---------- STEP 5: SAVE ----------
        if progress_callback:
            progress_callback(5, 5)

        joblib.dump(model, self.model_output)

        return {
            "accuracy": acc,
            "model_path": self.model_output,
        }
