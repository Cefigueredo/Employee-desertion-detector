import pydantic

from schemas import enums


class Employee(pydantic.BaseModel):
    Edad: int
    ViajesNegocio: enums.ViajesNegocioEnum
    PermaneceEnEmpresa: enums.PermanceEnEmpresaEnum
    Distancia_casa: int
    ID_empleado: int
    Satisfacción_ambiente: int
    Genero: enums.GeneroEnum
    Compromiso: int
    Satisfaccion_trabajo: int
    Estado_civil: enums.EstadoCivilEnum
    Ingreso_mensual: int
    SobreTiempo: enums.SobreTiempoEnum
    Horas_Produccion: int
    Bonos: int
    Años_trabajando: int
    Años_Compañia: int
    Años_Rol_Actual: int
    Años_Actual_Jefe: int
