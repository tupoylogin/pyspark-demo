London Taxi Trafic Simulation and Analytics
===========================================
A simple python script that simulates taxi trafic across London. Another simple script, written also in python, but with some usage of Spark, calculates simple analytics over generated dataset 

### Run Script
**Install missing packages**

```
pip install -r requirements.txt
```

**Initialize the taxi trafic dataset**

```
python faker.py "data/London postcodes.csv" "data/rides.csv" --num_rows=10_000_000
```

**Calculate simple some metrics**

```
python map_reduce.py
```
