import numpy as np
import pandas as pd

from src.predict import predict


class FakeModel:
    def predict(self, data):
        return np.array([1, 0])


def test_predict():

    data = pd.DataFrame(
        {
            "age": [50, 30],
            "bp": [90, 70],
        }
    )

    model = FakeModel()

    result = predict(model, data)

    assert list(result["prediction"]) == [1, 0]
    assert result.index.equals(data.index)
