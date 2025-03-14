# Gunakan image Python resmi sebagai base image
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements.txt ke working directory
COPY requirements.txt .

# Perbarui pip
RUN pip install --upgrade pip

# Install dependensi Python dengan mirror yang lebih cepat dan timeout yang lebih lama
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=1000 -r requirements.txt

# Copy seluruh isi proyek ke working directory
COPY . .

# Expose port 5000 untuk aplikasi Flask
EXPOSE 5000

# Perintah untuk menjalankan aplikasi Flask
CMD ["python", "app.py"]