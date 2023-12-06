# Employee desertion detector

I analyze the data from a company and its workers. I Take different variables into account like:

- Satisfacción_ambiente
- Satisfaccion_trabajo
- Estado_civil
- Ingreso_mensual
- SobreTiempo

Among others.

Then I create a FastAPI (Python) Application to deploy the Machine Learning Model and use it in an API REST.

Create a virtual environment with:

```bash
python -m venv venv
```

Activate the virtual environment with:

```bash
source venv/bin/activate
```

Run the application with:

```bash
uvicorn main:app --reload
```
