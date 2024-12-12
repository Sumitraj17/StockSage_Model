from fastapi import FastAPI, File, UploadFile
import uvicorn
import json

from predict_script import predict_sales;

# # Initialize FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Process the uploaded file and make predictions
    # print(file.file)
    json_result = predict_sales(file.file,"stocksage_model.pkl")
    return json_result

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)

# json_result = predict_sales("sales.csv","stocksage_model.pkl")