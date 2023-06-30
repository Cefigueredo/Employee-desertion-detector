import enum


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
