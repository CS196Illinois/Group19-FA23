
import model as m
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import database_operations as db_ops


app = FastAPI()

# Data model for the request body
class StockPredictionRequest(BaseModel):
    ticker: str
    n_future: int
    type: str

# Data model for the response body
class StockPredictionResponse(BaseModel):
    json_merged_df: str
    json_original_df: str

    # Converts the merged_df and original_df to json
    @classmethod
    def from_df_to_json(cls, merged_df: pd.DataFrame, original: pd.DataFrame) -> "StockPredictionResponse":
        cls.json_merged_df = merged_df.to_json(orient="split")
        cls.json_original_df = original.to_json(orient="split")
        return cls(json_merged_df=cls.json_merged_df, json_original_df=cls.json_original_df)
    
    # Converts the json_merged_df and json_original_df to a dictionary for response to the client
    def to_dict(self) -> dict[str, str]:
        return {
            "json_merged_df": self.json_merged_df,
            "json_original_df": self.json_original_df
        }

# Endpoint for the API
@app.post("/predict/")
async def predict_stock_price(request_data: StockPredictionRequest) -> dict[str, str]:
    ticker: str = request_data.ticker
    n_future: int = request_data.n_future
    type: str = request_data.type

    stock: m.NewStock | m.OldStock | None = None
    model: m.Sequential | None = None
    name: str = f'{ticker}_{type}_{n_future}.h5'

    in_db = db_ops.check_if_exists(name)
    if in_db:
        stock = m.OldStock(ticker, n_future, name)
        model = stock.get_model(name)
    else:
        stock = m.NewStock(ticker, n_future)
        model = stock.get_model(type)
    model = stock.fit_model(model)
    db_ops.save_model_to_db(model, name, in_db, stock.now_index)
    raw_pred = stock.predict(model, stock.test_data)
    merged_df = stock.reshape(raw_pred, stock.scaler, stock.scaled_data, stock.original, n_future)
    # last_prediction: float = merged_df['Predicted'].iloc[-1]

    response_data = StockPredictionResponse.from_df_to_json(merged_df=merged_df, original=stock.original)
    return response_data.to_dict()
