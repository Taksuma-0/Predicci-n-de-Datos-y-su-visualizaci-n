# Usa una imagen oficial de Python como base
FROM python:3.11-slim

# Establece la carpeta de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias al contenedor
COPY requirements.txt .

# Instala las dependencias dentro del contenedor
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos los archivos del proyecto (código, modelos, data)
COPY . .

# Comando que se ejecutará automáticamente al iniciar el contenedor
CMD ["streamlit", "run", "pagina.py", "--server.port=8501", "--server.address=0.0.0.0"]