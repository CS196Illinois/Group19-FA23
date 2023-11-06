
import model as m
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import database_operations as db_ops


app = FastAPI()

class StockPredictionRequest(BaseModel):
    ticker: str
    n_future: int

class StockPredictionResponse(BaseModel):
    json_merged_df: str
    json_original_df: str

    @classmethod
    def from_df_to_json(cls, merged_df: pd.DataFrame, original: pd.DataFrame):
        cls.json_merged_df = merged_df.to_json(orient="split")
        cls.json_original_df = original.to_json(orient="split")
        return cls(json_merged_df=cls.json_merged_df, json_original_df=cls.json_original_df)
    
    def to_dict(self):
        return {
            "json_merged_df": self.json_merged_df,
            "json_original_df": self.json_original_df
        }
    
@app.post("/predict/")
async def predict_stock_price(request_data: StockPredictionRequest):
    ticker = request_data.ticker
    n_future = request_data.n_future

    stock = None
    in_db = db_ops.check_if_exists(f'{ticker}.h5')
    if in_db:
        stock = m.OldStock(ticker, n_future)
    else:
        stock = m.NewStock(ticker, n_future)
    model = stock.get_model()
    model = stock.fit_model(model)
    db_ops.save_model_to_db(model, f'{ticker}.h5', in_db, stock.now_index)
    raw_pred = stock.predict(model, stock.test_data)
    merged_df = stock.reshape(raw_pred, stock.scaler, stock.scaled_data, stock.original)

    response_data = StockPredictionResponse.from_df_to_json(merged_df=merged_df, original=stock.original)
    return response_data.to_dict()