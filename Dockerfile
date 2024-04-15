FROM python:3.10

RUN pip install pandas
RUN pip install scikit-learn

CMD [ "python", "t.py" ]