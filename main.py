import pickle
from typing import Any

import fastapi
import pandas as pd

import cleaner
from schemas import employee

app = fastapi.FastAPI()

tree_model = pickle.load(open("models/tree_model.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model.pkl", "rb"))


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def post_predict(
    employee: employee.Employee = fastapi.Body(
        example={
            "Edad": 47,
            "ViajesNegocio": "Pocos",
            "PermaneceEnEmpresa": "SI",
            "Distancia_casa": 5,
            "ID_empleado": 447,
            "Satisfacción_ambiente": 4,
            "Genero": "Hombre",
            "Compromiso": 3.0,
            "Satisfaccion_trabajo": 3,
            "Estado_civil": "Casado",
            "Ingreso_mensual": 18300,
            "SobreTiempo": "No",
            "Horas_Produccion": 80,
            "Bonos": 1,
            "Años_trabajando": 21,
            "Años_Compañia": 3,
            "Años_Rol_Actual": 2.0,
            "Años_Actual_Jefe": 1,
        }
    )
) -> dict[str, Any]:
    df = pd.DataFrame(employee.dict(), index=[0])
    cleaned_input = cleaner.cleaner(df)
    y_pred = tree_model.predict(cleaned_input.iloc[[-1]])

    return {
        "prediction": "Permanecerá"
        if int(y_pred[0]) == 1
        else "No permanecerá"
    }
