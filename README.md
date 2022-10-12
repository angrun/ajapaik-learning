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

How to query data from ajapaik postgres DB
=======
```shell script
ssh -L 5430:127.0.0.1:5430 anna@ajapaik.ee -p 8022
sql -p 5430 -U rephoto_replica_ro -d rephoto_replica -h 127.0.0.1
= > SELECT ...
```
To query something into file
```shell script
ssh -L 5430:127.0.0.1:5430 anna@ajapaik.ee -p 8022
psql -p 5430 -U rephoto_replica_ro -d rephoto_replica -h 127.0.0.1 -c "select id, created, viewpoint_elevation,scene, user_id  from  project_photo LIMIT 100;" -o out.txt
scp -p 8022 anna@ajapaik.ee:~/out.txt /Users/annagrund/Desktop/
```
