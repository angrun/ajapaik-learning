Ajapaik learning
=======

How to run
=======

From /app folder run:

```shell script
python3.7 manage.py runserver 7000
```

```shell script
curl -X GET localhost:7000/predict/test_2.jpg


[0.91004425 0.0899557 ] // [exterior_probability, interior_probablity]
```