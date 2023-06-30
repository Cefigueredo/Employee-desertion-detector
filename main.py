import pandas as pd
import enum
import pickle
import cleaner
import fastapi
import json
import pydantic
from typing import Any

app = fastapi.FastAPI()

tree_model = pickle.load(open("models/tree_model.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model.pkl", "rb"))


class ViajesNegocioEnum(str, enum.Enum):
    NUNCA = "Nunca"
    POCOS = "Pocos"
    FRECUENTES = "Frecuentes"


class GeneroEnum(str, enum.Enum):
    HOMBRE = "Hombre"
    MUJER = "Mujer"


class EstadoCivilEnum(str, enum.Enum):
    SOLTERO = "Soltero"
    CASADO = "Casado"
    DIVORCIADO = "Divorciado"


class SobreTiempoEnum(str, enum.Enum):
    SI = "Si"
    NO = "No"


class PermanceEnEmpresaEnum(str, enum.Enum):
    SI = "SI"
    NO = "NO"


class Employee(pydantic.BaseModel):
    Edad: int
    ViajesNegocio: ViajesNegocioEnum
    PermaneceEnEmpresa: PermanceEnEmpresaEnum
    Distancia_casa: int
    ID_empleado: int
    Satisfacción_ambiente: int
    Genero: GeneroEnum
    Compromiso: int
    Satisfaccion_trabajo: int
    Estado_civil: EstadoCivilEnum
    Ingreso_mensual: int
    SobreTiempo: SobreTiempoEnum
    Horas_Produccion: int
    Bonos: int
    Años_trabajando: int
    Años_Compañia: int
    Años_Rol_Actual: int
    Años_Actual_Jefe: int


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def post_predict(
    employee: Employee = fastapi.Body(
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
