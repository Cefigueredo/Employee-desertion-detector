import numpy as np
import pandas as pd


def cleaner(df: pd.DataFrame):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df.drop(["ID_empleado"], axis=1)
    df = df.drop(["Horas_Produccion"], axis=1)
    df["ViajesNegocio"].replace("Ninguno", "Nunca", inplace=True)
    df["Satisfacción_ambiente"].replace(5, 4, inplace=True)
    df["Genero"].replace("M", "Hombre", inplace=True)
    df["Genero"].replace("F", "Mujer", inplace=True)

    atributo = "ViajesNegocio"

    def label_categorias_viajes(row):
        if row[atributo] == "Nunca":
            return 0
        elif row[atributo] == "Pocos":
            return 1
        elif row[atributo] == "Frecuentes":
            return 2
        else:
            return None

    df[atributo] = df.apply(lambda row: label_categorias_viajes(row), axis=1)

    atributo = "Genero"

    def label_categorias_genero(row):
        if row[atributo] == "Mujer":
            return 1
        elif row[atributo] == "Hombre":
            return 0
        else:
            return None

    df[atributo] = df.apply(lambda row: label_categorias_genero(row), axis=1)

    atributo = "Estado_civil"

    def label_categorias_estado_civil(row):
        if row[atributo] == "Soltero":
            return 0
        elif row[atributo] == "Casado":
            return 1
        elif row[atributo] == "Divorciado":
            return 2
        else:
            return None

    df[atributo] = df.apply(
        lambda row: label_categorias_estado_civil(row), axis=1
    )

    atributo = "SobreTiempo"

    def label_categorias_sobretiempo(row):
        if row[atributo] == "Si":
            return 1
        elif row[atributo] == "No":
            return 0
        else:
            return None

    df[atributo] = df.apply(
        lambda row: label_categorias_sobretiempo(row), axis=1
    )

    number_cols = df.dtypes[
        (df.dtypes == np.int64) | (df.dtypes == np.float64)
    ].index
    number_cols = df.select_dtypes(include=["int64", "float"]).columns

    cleaned_df = df[number_cols]

    return cleaned_df.iloc[0:2]
