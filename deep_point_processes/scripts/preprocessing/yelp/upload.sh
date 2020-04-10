#!/usr/bin/env bash

DB="yelp"

for i in ./*.json; do
    fileName=${i##*/}
    collectionName=${fileName%.json}
    mongoimport --db $DB --collection $collectionName --file $fileName --drop
done

mongo yelp.js