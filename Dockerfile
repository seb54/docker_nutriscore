FROM python:3.13.0-slim

WORKDIR /app

COPY model.pkl .
COPY scaler.pkl .
COPY cleaned_data.csv .
COPY api.py .

RUN pip install flask numpy scikit-learn pandas

EXPOSE 5000

CMD ["python", "api.py"]