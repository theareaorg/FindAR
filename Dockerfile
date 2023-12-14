from python
COPY area area
COPY setup.py setup.py
COPY run.py run.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install .
RUN mkdir storage